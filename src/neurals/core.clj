(ns neurals.core)

;; this is the function that is our circuit.
;; the question is that :
;;     given some (x, y) we get val <- forward-multiply(x, y).
;;     how we tweak x and y a little so that we get new val2 > val.
(defn forward-multiply
  "Multiply two numbers"
  [^:Int x ^:Int y]
  (* x y))

(defn change-little
  "Change the values (x and y) a little."
  [^:Int x ^:Ratio r]
  (+ x (* r (- (* 2 (rand)) 1)))
  ;;(+ x (* r (rand)))
  )

;; strategy-1.
(defn local-search-better-forward
  "Finds the best forward-multiply using local-search technique. 
  Local-search is random search around the points (x, y)."
  [^:Int x ^:Int y ^:Int times]
  (let [tweak-amount 0.01]
    (->> (repeat [x y])
         (map (fn [[x y]]
                [(change-little x tweak-amount) 
                 (change-little y tweak-amount)]))
         (map (fn [[x y]] {:val (forward-multiply x y)
                           :x x
                           :y y}))
         (take times)
         (reduce (fn [valx valy] (if (> (:val valx) (:val valy))
                                   valx
                                   valy))))))


(defn- derivative-forward 
  "Derivative of the forward-multiply function at values x and y 
  with change chg-x and chg-y"
  [x y chg-x chg-y h]
  (let [org-val (forward-multiply x y)
        chg-val (forward-multiply chg-x chg-y)
        change (- chg-val org-val)]
    (/ change h)))

(defn x-derivative-forward 
  "Gives the x-gradient"
  [x y h]
  (derivative-forward x y (+ x h) y h))

(defn y-derivative-forward 
  "Gives the y-gradient"
  [x y h]
  (derivative-forward x y x (+ y h) h))

(defn numerical-gradient-forward
  "Numerical Gradient: Good method to find the better (higher) value of
  forward-multiply() at point (x, y). It computes the derivates by calculating
  slope at this point."
  [x y]
  (let [step 0.01
        h 0.0001
        chg-x (+ x (* step (x-derivative-forward x y h)))
        chg-y (+ y (* step (y-derivative-forward x y h)))]
    {:val (forward-multiply chg-x chg-y)
     :x chg-x
     :y chg-y}))

(defn analytical-gradient-forward
  "Analytic Gradient: Best method. As we gets the derivative by calculus and hence,
  we do need to just calculate our forward-multiply()."
  [x y]
  (let [step 0.01
        chg-x (+ x (* step y)) ;; using calculus find the x-derivative 
        ;; and y-derivative which is y and x respectively.
        chg-y (+ y (* step x))]
    {:val (forward-multiply chg-x chg-y)
     :x chg-x
     :y chg-y}))

;; Summary Till now :
;; To compute the gradient we went from forwarding the circuit hundreds of times
;; (Strategy #1) to forwarding it only on order of number of times twice the 
;; number of inputs (Strategy #2), to forwarding it a single time! And it gets
;; EVEN better, since the more expensive strategies (#1 and #2) only give an
;; approximation of the gradient, while #3 (the fastest one by far) gives you the
;; exact gradient. No approximations. The only downside is that you should be
;; comfortable with some calculus 101.

;; ------------------------------------------------------------------------------------

;; ----- Gates -----

;; A single unit is the input-to-the-circuit.
(defrecord Unit 
    [id name]
  Object
  (toString [_]
    (str name " : ")))

;; A Gate has two units of inputs. 
(defrecord Gate
    [^:Unit input-a ^:Unit input-b])

(defprotocol GateOps
  "Basic Gate Operations: Forward and Backward are two 
  protocol-operations need to be supported by  each gate."
  (forward [this _] "Give the output-value from input gate(s) used in 
going forward the circuit. ")
  (backward [this back-grad init-values] "Gives the gradient to its 
input - argument has back-grad : which is gradient from its output. 
input-gates : stores the input given to the circuit.
Backward calcuates the derivative for generating the backward pull."))

;; Unit is mouth of cirtuit and hence simple operations.
(extend-protocol GateOps
  Unit
  (forward [this input-values]
    (let [id (:id this)]
      {id (input-values id)}))
  (backward [this back-grad init-values]
    {(:id this) back-grad}))

;; MultiplyGate gets two inputs and * their values going forward.
(defrecord MultiplyGate [id input-a input-b]
  GateOps
  (forward [this input-values]
      (let [id (:id this)
        input-a (:input-a this)
        input-b (:input-b this)
        val-a-map (forward input-a input-values)
        val-b-map (forward input-b input-values)]
    (-> 
     (merge-with + val-a-map val-b-map)
     (assoc   id (* (val-a-map (:id input-a)) 
                    (val-b-map (:id input-b)))))))

  (backward [this back-grad init-values]
    (let [input-a (:input-a this)
          input-b (:input-b this)
          val-a (init-values (:id input-a))
          val-b (init-values (:id input-b))]
      (-> 
       (merge-with + (backward input-a (* val-b back-grad) init-values)
                   (backward input-b (* val-a back-grad) init-values))
       (assoc  (:id this) back-grad)))))

;; AddGate add values of two  inputs.
(defrecord AddGate [id input-a input-b]
  GateOps
  (forward [this input-values]
      (let [id (:id this)
        input-a (:input-a this)
        input-b (:input-b this)
        val-a-map (forward input-a input-values)
        val-b-map (forward input-b input-values)]
    (-> 
     (merge-with + val-a-map val-b-map)
     (assoc   id (+ (val-a-map (:id input-a)) 
                    (val-b-map (:id input-b)))))))

  (backward [this back-grad init-values]
    (let [input-a (:input-a this)
          input-b (:input-b this)]
      (-> 
       (merge-with + (backward input-a (* 1.0 back-grad) init-values)
                   (backward input-b (* 1.0 back-grad) init-values))
       (assoc   (:id this) back-grad)))))


(defn sig 
  "Sigmoid function : f(x) = 1/(1 + exp(-x))"
  [x]
  (/ 1 (+ 1 (Math/pow Math/E (- x)))))

;; SigmoidGate applies sig on input.
(defrecord SigmoidGate [id gate]
  GateOps
  (forward [this input-values]
      (let [id (:id this)
        input-a (:gate this)
        val-a-map (forward input-a input-values)]
    (-> 
     val-a-map
     (assoc   id (sig (val-a-map (:id input-a)))))))

  (backward [this back-grad init-values]
    (let [s (init-values (:id this))
          ;; s is (sig input) i.e. output
          ds (* s (- 1 s) back-grad)]
      (->
       (backward (:gate this) ds init-values)
       (assoc (:id this) back-grad)))))


(defmacro defunit 
  "Creates a Unit that also stores the name of the variable."
  [var-name body]
  `(def ~var-name (~@body (name '~var-name))))

;; neural network : f(x,y) = sig(a*x + b*y + c)
(defunit a (->Unit 0))
(defunit b (->Unit 1))
(defunit c (->Unit 2))
(defunit x (->Unit 3))
(defunit y (->Unit 4))

(def ax (->MultiplyGate 5 a x))
(def by (->MultiplyGate 6 b y))
(def axc (->AddGate 7 ax c))
(def axcby (->AddGate 8 axc by))
(def sigaxcby (->SigmoidGate 9 axcby ))

(clojure.pprint/pprint 
 (backward sigaxcby 1.0 
           ;; forward takes a map { unit-id  input-to-unit }
           (forward sigaxcby {0 1.0
                              1 2.0
                              2 -3.0
                              3 -1.0
                              4 3.0
                              })))

;; --------------- *** ---------------------------

