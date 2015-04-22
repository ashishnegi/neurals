(ns neurals.multi
  (:require [neurals.core :as core]))

;; Plan : 1. use multimethods to do the same thing as above.
;; Plan : 2. create functions rather than objects for Gates. Think about a 
;;        better way to solve the problem of `backward` in protocols.
;;        `backward` is costly above without immutability.

;; Plan 1 : Execution :)

(defmulti forward (fn [x _] (:type x)))
(defmulti backward (fn [x _ _] (:type x)))

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
  {this back-grad})

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


(clojure.pprint/pprint 
 (backward sigaxcby 1.0 
           ;; forward takes a map { unit-id  input-to-unit }
           (forward sigaxcby {0 1.0
                              1 2.0
                              2 -3.0
                              3 -1.0
                              4 3.0
                              })))


;; Plan 1 : DONE :)

;; ---------------- **** -----------------------------
