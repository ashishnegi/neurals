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

(defn addition-gate [gate-a gate-b]
  {:type :add-gate
   :gate-a gate-a
   :gate-b gate-b
   :id (gensym "id")})

(defn mul-gate [gate-a gate-b] 
  {:type :mul-gate
   :gate-a gate-a
   :gate-b gate-b
   :id (gensym "id")})

(defn sig-gate [gate-a]
  {:type :sig-gate
   :gate-a gate-a
   :id (gensym "id")})

(defn op-on-gates [op this]
  (op (forward (:gate-a this))
      (forward (:gate-b this))))

(defmethod forward :unit
  [this input-values]
  (let [id (:id this)]
    {id (input-values id)}))

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

(defmethod backward :unit
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

;; f(x,y) = a*x + b*y + c

(defunit a 0) ;; 0 is the id for a
(defunit b 1)
(defunit c 2)
(defunit x 3)
(defunit y 4)

(def ax (mul-gate a x))
(def by (mul-gate b y ))
(def axc (addition-gate ax c))
(def axcby (addition-gate axc by))
(def sigaxcby (sig-gate axcby))

(def initial-data  {0 1.0
                    1 2.0
                    2 -3.0
                    3 -1.0
                    4 3.0
                    })

(clojure.pprint/pprint 
 (backward sigaxcby 1.0 
           ;; forward takes a map { unit-id  input-to-unit }
           (forward sigaxcby initial-data)))


;; Plan 1 : DONE :)

;; ---------------- **** -----------------------------
(defn make-better-value [neuron, input]
  (loop [input input]
    (let [id-neuron (:id neuron)
          aaaz (clojure.pprint/pprint input)
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
          aaaf (clojure.pprint/pprint (- val-neuron-new val-neuron-old))]
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
  (loop [input initial-data]
    (let [values (forward neuron input)
          id-neuron (:id neuron)
          val-neuron (values id-neuron)
          pull (find-pull label val-neuron)
          
          all-gates (inputs neuron)
          ;; ids of the input (mouth) of circuit.
          input-ids (set (map #(:id %1) 
                              (filter #(= :unit (:type %1)) all-gates)))
          
          gradients (if (zero? pull)
                      ;; fill the gradients with 0s.
                      ;; (reduce (fn [x y] (assoc x y 0)) {} input-ids)
                      ;; otherwise calculate backward gradient.
                      (backward neuron pull values)
                      (backward neuron pull values))

          ;; aaaa (clojure.pprint/pprint gradients)
          ;; aaab (clojure.pprint/pprint pull)

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
      new-input)))

(defunit a1 0) ;; 0 is the id for a
(defunit b1 1)
(defunit c1 2)
(defunit x1 3)
(defunit y1 4)

(def ax1 (mul-gate a1 x1))
(def by1 (mul-gate b1 y1))
(def ax-plus-c (addition-gate ax1 c1))
(def svm-neuron (addition-gate ax-plus-c by1))

(defn cal-input-ab-op [step grad tozero]
  (* step (+ grad tozero)))

(defn cal-input-c-op [step grad tozero]
  (* step grad))

(defn cal-input-xy-op [step grad tozero]
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

(defn make-svm-input [coeff-ins xy-s]
  (merge coeff-ins {3 (first xy-s)
                    4 (second xy-s)
                    :label (nth xy-s 2)}))

(defn svm-learn [times]
  (loop [t times 
         input (list (make-svm-input svm-init-data (first xy-label-s)))]
    (if (< t 0)
      input
      (do
        ;; (clojure.pprint/pprint input)
        ;; (let [data (first input)
        ;;       label ((forward svm-neuron data) 
        ;;                           (:id svm-neuron))]
        ;;   (clojure.pprint/pprint label)
        ;;   (clojure.pprint/pprint (find-pull (:label data) label)))
        (recur (dec t) 
               (concat  
                     (reverse 
                      (reductions 
                       (fn [new-input xy-label]
                         (let [data (make-svm-input new-input xy-label)]
                           (do         
                             ;; (clojure.pprint/pprint data)
                             (svm-new-input svm-neuron 
                                            data
                                            (:label data)
                                            cal-input-ops))))
                       (first input) xy-label-s))
                     input))))))

;; I am getting 100% accuracy even from 1st complete iteration.
;; I think some problems are there..
(defn svm-accuracy [times]
  (->> (first (svm-learn times))
       (repeat)
       (map (fn [xy-label data]
              (make-svm-input data xy-label)) xy-label-s)
       (map (fn [data] 
              (let [values (forward svm-neuron data)
                    id-neuron (:id svm-neuron)
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


(defn find-pull [label val-neuron]
  (if (and (< label 0)
           (> val-neuron label))
    -1.0
    (if (and (> label 0)
             (< val-neuron label))
      1.0
      0)))
