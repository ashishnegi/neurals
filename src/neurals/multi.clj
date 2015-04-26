(ns neurals.multi
  (:require [neurals.core :as core]))

;; Plan : 1. use multimethods to do the same thing as above.
;; Plan : 2. create functions rather than objects for Gates. Think about a 
;;        better way to solve the problem of `backward` in protocols.
;;        `backward` is costly above without immutability.

;; Plan 1 : Execution :)

(defmulti forward (fn [x _] (:type x)))
(defmulti backward (fn [x _ _] (:type x)))
(defmulti inputs (fn [x] (:type x)))

(defmacro defunit [var-name id]
  `(def ~var-name {:type :unit 
                   :name (name '~var-name)
                   :id ~id}))

(defn addition-gate
  "Makes a Add gate. Adds the output value of two gates."
  [gate-a gate-b]
  {:type :add-gate
   :gate-a gate-a
   :gate-b gate-b
   :id (gensym "id")})

(defn mul-gate 
  "Makes a Multiply Gate."
  [gate-a gate-b] 
  {:type :mul-gate
   :gate-a gate-a
   :gate-b gate-b
   :id (gensym "id")})

(defn sig-gate
  "Makes the sigmoid gate."
  [gate-a]
  {:type :sig-gate
   :gate-a gate-a
   :id (gensym "id")})

(defn max0-gate
  "Makes the max0-gate which outputs only positive values.
  If input > 0, input is outputted, else 0."
  [gate-a]
  {:type :max0-gate
   :gate-a gate-a
   :id (gensym "id")})

(defn op-on-gates [op this]
  (op (forward (:gate-a this))
      (forward (:gate-b this))))

;; define the forward for all the <:type> of gates.
(defmethod forward :unit
  [this input-values]
  (let [id (:id this)]
    (do ;; (clojure.pprint/pprint {id (input-values id)})
        {id (input-values id)})))

(defmethod forward :add-gate
  [this input-values]
  (let [id (:id this)
        gate-a (:gate-a this)
        gate-b (:gate-b this)
        val-a-map (forward gate-a input-values)
        val-b-map (forward gate-b input-values)]
    (-> 
     (merge-with + val-a-map val-b-map)
     (assoc   id (+ (val-a-map (:id gate-a)) 
                    (val-b-map (:id gate-b)))))))

(defmethod forward :mul-gate
  [this input-values]
  (let [id (:id this)
        gate-a (:gate-a this)
        gate-b (:gate-b this)
        val-a-map (forward gate-a input-values)
        val-b-map (forward gate-b input-values)]
    (-> 
     (merge-with + val-a-map val-b-map)
     (assoc   id (* (val-a-map (:id gate-a)) 
                    (val-b-map (:id gate-b)))))))

(defmethod forward :sig-gate
  [this input-values]
  (let [id (:id this)
        gate-a (:gate-a this)
        val-a-map (forward gate-a input-values)]
    (-> 
     val-a-map
     (assoc   id (core/sig (val-a-map (:id gate-a)))))))

(defmethod forward :max0-gate
  [this input-values]
  (let [id (:id this)
        gate-a (:gate-a this)
        val-a-map (forward gate-a input-values)]
    (-> val-a-map
        (assoc   id (max (val-a-map (:id gate-a)) 0)))))

;; ---- forward of gates done ---------

;; define the backward on all gates.
(defmethod backward :unit
  [this back-grad values-gates]
  {(:id this) back-grad})

(defmethod backward :variable
  [this back-grad values-gates]
  {(:id this) back-grad})

(defmethod backward :add-gate
  [this back-grad values-gates]
  (let [gate-a (:gate-a this)
        gate-b (:gate-b this)]
    (-> 
     (merge-with + (backward gate-a (* 1.0 back-grad) values-gates)
                 (backward gate-b (* 1.0 back-grad) values-gates))
     (assoc (:id this) back-grad))))

(defmethod backward :mul-gate
  [this back-grad values-gates]
  (let [gate-a (:gate-a this)
        gate-b (:gate-b this)
        val-a (values-gates (:id gate-a))
        val-b (values-gates (:id gate-b))]
    (-> 
     (merge-with + (backward gate-a (* val-b back-grad) values-gates)
                 (backward gate-b (* val-a back-grad) values-gates))
     (assoc   (:id this) back-grad))))

(defmethod backward :sig-gate
  [this back-grad values-gates]
  (let [gate-a (:gate-a this)
        s (values-gates (:id this))]
    (-> 
     (backward gate-a (* s (- 1 s) back-grad) values-gates)
     (assoc  (:id this) back-grad))))

(defmethod backward :max0-gate
  [this back-grad values-gates]
  (let [id (:id this)
        gate-a (:gate-a this)
        val-a (values-gates (:id gate-a))
        val-b 0
        a-gt-b (> val-a val-b)
        pull-a (if a-gt-b 1 0)]
    (-> (backward gate-a (* pull-a back-grad) values-gates)
        (assoc id back-grad))))

;; ----- backward on gates done. --------

;; define the inputs on the gates.
(defmethod inputs :unit
   [this]
   [this])

(defmethod inputs :add-gate
  [this]
  (conj (concat (inputs (:gate-a this))
                (inputs (:gate-b this)))
        this))

(defmethod inputs :mul-gate
  [this]
  (conj (concat (inputs (:gate-a this))
                (inputs (:gate-b this)))
        this))

(defmethod inputs :sig-gate
  [this]
  (conj (inputs (:gate-a this)) this))

(defmethod inputs :max0-gate
  [this]
  (conj (inputs (:gate-a this)) this))

;; ---- end : inputs on gates done. ------

;; simple Neuron 
;; f(x,y) = sig(a*x + b*y + c)

(defunit a 0) ;; 0 is the id for a
(defunit b 1)
(defunit c 2)
(defunit x 3)
(defunit y 4)

(def ax (mul-gate a x))
(def by (mul-gate b y ))
(def axc (addition-gate ax c))
(def axcby (addition-gate axc by))
;; this is f(x,y)
(def sigaxcby (sig-gate axcby)) 

(def initial-data  {0 1.0
                    1 2.0
                    2 -3.0
                    3 -1.0
                    4 3.0
                    })

;; Plan 1 : DONE :)
;; Plan 2 : Also done :).
;; ---------------- **** -----------------------------
(defn make-better-value 
  "Keep generating new-values by pulling the circuit,
  untill we are not able to make any appreciable progress."
  [neuron, input]
  (loop [input input]
    (let [id-neuron (:id neuron)
          ;; aaaz (clojure.pprint/pprint input)
          values (forward neuron input)
          gradients (backward neuron 1.0 values)
          step 0.01
          all-gates (inputs neuron)
          input-ids (set (map #(:id %1) 
                              (filter #(= :unit (:type %1)) all-gates)))
          ;; aaaa (clojure.pprint/pprint input-ids)
          ;; aaad (clojure.pprint/pprint values)
          ;; aaac (clojure.pprint/pprint gradients)

          ;; create new input from gradients on units, and 
          ;; initial `input` on units.
          val-gradient-pairs (filter (fn [x]
                                       (input-ids (first x)))
                                     (merge-with (fn [x y]
                                                   [x y]) 
                                                 input gradients))
          ;; aaab (clojure.pprint/pprint val-gradient-pairs)
          new-input (reduce (fn [new-map v]
                              (let [key (first v)
                                    value (first (second v))
                                    gradient (second (second v))]
                                (assoc new-map key
                                       (+ value (* step gradient)))))
                            {} val-gradient-pairs)
          ;; aaae (clojure.pprint/pprint new-input)
          new-values (forward neuron new-input)
          val-neuron-old (values id-neuron)
          val-neuron-new (new-values id-neuron)
          better-threshold 0.0004
          ;; aaaf (clojure.pprint/pprint (- val-neuron-new val-neuron-old))
          ]
      (if (< (- val-neuron-new val-neuron-old) better-threshold)
        val-neuron-new
        (recur new-input)))))

;; should be able to make better-output 
;; with back-and-forward propagation.
(> (make-better-value sigaxcby initial-data) 
   ((forward sigaxcby initial-data) (:id sigaxcby)))
;; ------------- *** ------------------------------

;; so Above in make-better-value, we changed [a,b,c] and [x,y].
;; where [a,b,c] were the coefficients and [x,y] were variables
;; in f(x) = a*x + b*y + c

;; 1. Need to decide how to send new-val-calc operation to SVD.
;;    Currently sending as separate argument.

(declare find-pull)

;; In SVM, we only apply pull on [a,b,c].
;; Also, we apply additional pull on [a,b] to take it towards 0.
(defn svm-new-input 
  "1. calculate forwad values on input.
   2. test if matches the label or not ?
   3.  if matches then pull is 0 -> no backward gradient.
   4.  else pull is 1/-1 -> calculate backward gradients.
   5. calculate new values with (input,backward-gradients,to-zero].
   6. to calculate new-values, find units and call new-val-calc-ops."
  ;; label -> {-1,1} only.
  [neuron initial-data label new-val-calc-ops]
  (let [input initial-data
        values (forward neuron input)
        id-neuron (:id neuron)
        val-neuron (values id-neuron)
        pull (find-pull label val-neuron)
        
        all-gates (inputs neuron)
        ;; ids of the input (mouth) of circuit.
        input-ids (set (map #(:id %1) 
                            (filter #(= :unit (:type %1)) all-gates)))
        
        gradients (if (zero? pull)
                    ;; fill the gradients with 0s.
                    (reduce (fn [x y] (assoc x y 0)) {} input-ids)
                    ;; otherwise calculate backward gradient.
                    (backward neuron pull values))

        ;; aaaa (clojure.pprint/pprint pull)
        ;; aaab (clojure.pprint/pprint label)
        ;; aaac (clojure.pprint/pprint input-ids)
        ;; aaad (clojure.pprint/pprint gradients)
        ;; aaae (clojure.pprint/pprint values)
        ;; aaaf (clojure.pprint/pprint input)

        ;; for each input-id, call its operation with
        ;; val,gradient,to-zero value of it.
        ;; [a,b] should work on them.
        ;; [c] should not work on to-zero.
        ;; [x] should just not change itself.
        step 0.01
        new-input (reduce 
                   (fn [new-map input-id]
                     (let [op (new-val-calc-ops input-id)
                           gradient-of-id (gradients input-id)
                           tozero-of-id (- (input input-id))
                           input-of-id (input input-id)]
                       (assoc new-map input-id 
                              (+ input-of-id (op step
                                                 gradient-of-id
                                                 tozero-of-id)))))
                   {}  input-ids)]
    new-input))

(defunit a1 0) ;; 0 is the id for a
(defunit b1 1)
(defunit c1 2)
(defunit x1 3)
(defunit y1 4)

(def ax1 (mul-gate a1 x1))
(def by1 (mul-gate b1 y1))
(def ax-plus-c (addition-gate ax1 c1))
(def svm-neuron (addition-gate ax-plus-c by1))

(defn cal-input-ab-op 
  "Operation for variable coefficient for generating new gradient."
  [step grad tozero]
  (* step (+ grad tozero)))

(defn cal-input-c-op 
  "Operation for constant coefficient."
  [step grad tozero]
  (* step grad))

(defn cal-input-xy-op 
  "Operation for variables in svm. No change."
  [step grad tozero]
  0)

(def cal-input-ops {0 cal-input-ab-op
                    1 cal-input-ab-op
                    2 cal-input-c-op
                    3 cal-input-xy-op
                    4 cal-input-xy-op
                    })

;; init-data only for [a,b,c].
;; xy would be assoced later.
(def svm-init-data {0 0
                    1 0
                    2 0
                    })

(def xy-label-s [[1.2 0.7 1]
                 [-0.3 -0.5 -1]
                 [3.0 0.1 1]
                 [-0.1 -0.1 -1]
                 [-1.0 1.1 -1]
                 [2.1 -3 1]])

;; Question : why not add regularization (tozero) to c ?
;; Question : why not change [x,y] ? of-course no point, since
;;            label depend upon them.

(defn make-svm-input 
  "Make svm-input from coeff-data and value of [x,y].
  [x,y]'s data is [3,4]."
  [coeff-ins xy-s]
  (merge coeff-ins {3 (first xy-s)
                    4 (second xy-s)
                    :label (nth xy-s 2)}))

(defn svm-learn 
  "Randomly learn the neuron for times.
  Data is xy-label-s."
  [times neuron init-data make-svm-input-op cal-input-ops]
  (let [t times 
        input init-data
        random-xy-s (repeatedly times #(rand-nth xy-label-s))]
    ;; (clojure.pprint/pprint input)
    ;; (let [data (first input)
    ;;       label ((forward neuron data) 
    ;;                           (:id neuron))]
    ;;   (clojure.pprint/pprint label)
    ;;   (clojure.pprint/pprint (find-pull (:label data) label)))
      
    (reduce
     (fn [new-input-list xy-label]
       (let [new-input (first new-input-list)
             data (make-svm-input-op new-input xy-label)]
         (do         
           ;; (clojure.pprint/pprint data)
           (conj new-input-list
                 (make-svm-input-op
                  (svm-new-input neuron 
                                 data
                                 (:label data)
                                 cal-input-ops)
                  xy-label)))))
     (list input) random-xy-s)))


(defn svm-accuracy 
  "Find the accuracy of the learning of neuron."
  [times neuron init-data make-svm-input-op cal-input-ops]
  (->> (first (svm-learn times neuron init-data make-svm-input-op
                         cal-input-ops))
       (repeat)
       (map (fn [xy-label data]
              (make-svm-input-op data xy-label)) xy-label-s)
       (map (fn [data] 
              (let [values (forward neuron data)
                    id-neuron (:id neuron)
                    val-neuron (values id-neuron)
                    label (:label data)]
                (if (or  (and (> label 0)
                              (> val-neuron 0))
                         (and (< label 0)
                              (< val-neuron 0)))
                  {:data data
                   :accurate true
                   :label-val val-neuron}
                  {:data data
                   :accurate false
                   :label-val val-neuron}))))
       ((fn [learn]
          (assoc {:success (count 
                            (filter #(= (:accurate %1) true) learn))
                  :total (count learn)}
                 :learn learn)))))

(defn find-pull 
  "Find pull for the svm.
  Argument label : label of input-data.
  Argument val-neuron : label generated from the circuit for input-data.
  Gives what should happen to the circuit for this val-neuron.
  If we are deviating from label, then punish the circuit by pulling it 
  in opposite direction."
  [label val-neuron]
  (if (and (< label 0)
           (> val-neuron label))
    -1.0
    (if (and (> label 0)
             (< val-neuron label))
      1.0
      0)))

;; ---------- **** -------------------------------------------------
;; 2-layer Neural Network with 3 hidden neurons (n1, n2, n3) that
;; uses Rectified Linear Unit (ReLU) non-linearity 
;; on each hidden neuron
;; f1(x,y) = max(0, a1*x + b1*y + c1)
;; f2(x,y) = max(0, a2*x + b2*y + c2)
;; f3(x,y) = max(0, a3*x + b3*y + c3)
;; f(x,y)  = f1(x,y)*x + f2(x,y)*y + f3(x,y)*c + d
(defunit x-0 1)
(defunit y-0 2)

(defunit a-1 3)
(defunit b-1 4)
(defunit c-1 5)
(def a1x (mul-gate a-1 x-0))
(def b1y (mul-gate b-1 y-0))
(def a1xc (addition-gate a1x c-1))
(def a1xcby (max0-gate (addition-gate a1xc b1y))) ;; f1(x,y)

(defunit a-2 6)
(defunit b-2 7)
(defunit c-2 8)
(def a2x (mul-gate a-2 x-0))
(def b2y (mul-gate b-2 y-0))
(def a2xc (addition-gate a2x c-2))
(def a2xcby (max0-gate (addition-gate a2xc b2y))) ;; f2(x,y)

(defunit a-3 9)
(defunit b-3 10)
(defunit c-3 11)
(def a3x (mul-gate a-3 x-0))
(def b3y (mul-gate b-3 y-0))
(def a3xc (addition-gate a3x c-3))
(def a3xcby (max0-gate (addition-gate a3xc b3y))) ;; f3(x,y)

(defunit a-4 12)
(defunit b-4 13)
(defunit c-4 14)
(defunit d-4 15)

;; Now x is a1xcby
;;     y is a2xcby
;;     z is a3xcby

(def a4x (mul-gate a-4 a1xcby))
(def b4y (mul-gate b-4 a2xcby))
(def c4z (mul-gate c-4 a3xcby))
(def a4xb4y (addition-gate a4x b4y))
(def a4xb4yc4z (addition-gate a4xb4y c4z))
(def neuron-2-stage (addition-gate a4xb4yc4z d-4)) ;; f(x,y)

(defn make-svm-input-2-layer [coeff-ins xy-label]
  (merge coeff-ins {
                    1 (first xy-label)
                    2 (second xy-label)
                    :label (nth xy-label 2)}))

(def cal-input-ops-neuron {
                           1 cal-input-xy-op
                           2 cal-input-xy-op
                           3 cal-input-ab-op
                           4 cal-input-ab-op
                           5 cal-input-c-op
                           6 cal-input-ab-op
                           7 cal-input-ab-op
                           8 cal-input-c-op
                           9 cal-input-ab-op
                           10 cal-input-ab-op
                           11 cal-input-c-op
                           12 cal-input-ab-op
                           13 cal-input-ab-op
                           14 cal-input-c-op
                           15 cal-input-c-op
                           })

(clojure.pprint/pprint 
  (svm-accuracy 67
             neuron-2-stage
             (reduce (fn [x y]
                       (assoc x y (+ 0.5 (rand -1))))
                     {} (range 0 16))
             make-svm-input-2-layer
             cal-input-ops-neuron))

;; learned important thing that - 
;; in svm-neuron since, c has + operator,
;; it would be able to get non-zero gradient in first step.
;; this would initiate the learning. however, in neuron-2-stage.
;; having 0 initial values for all, would not progress the neuron.
(clojure.pprint/pprint 
  (svm-accuracy 2
                svm-neuron 
                (reduce (fn [x y]
                          (assoc x y 0))
                        {} (range 0 6))
                make-svm-input
                cal-input-ops))
