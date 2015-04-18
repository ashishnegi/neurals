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


(defn derivative-forward 
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

;; ------------------------------------------------------------------------------------
