(ns neurals.multi
  (:require [neurals.core :as core]))

;; Plan : 1. use multimethods to do the same thing as above.
;; Plan : 2. create functions rather than objects for Gates. Think about a 
;;        better way to solve the problem of `backward` in protocols.
;;        `backward` is costly above without immutability.

;; Plan 1 : Execution :)

(defmulti forward (fn [x] (:type x)))
(defmulti backward (fn [x _] (:type x)))

(defmacro defunit [var-name value]
  `(def ~var-name {:type :unit 
               :value ~value
               :name (name '~var-name)}))

(defmethod forward :unit
  [this]
  (:value this))

(defmethod backward :unit
  [this back-grad]
  {this back-grad})

(defn addition-gate [gate-a gate-b]
  {:type :add-gate
   :gate-a gate-a
   :gate-b gate-b})

(defn mul-gate [gate-a gate-b] 
  {:type :mul-gate
   :gate-a gate-a
   :gate-b gate-b})

(defn sig-gate [gate-a]
  {:type :sig-gate
   :gate-a gate-a})

(defn op-on-gates [op this]
  (op (forward (:gate-a this))
      (forward (:gate-b this))))

(defmethod forward :add-gate
  [this]
  (op-on-gates + this))

(defmethod forward :mul-gate
  [this]
  (op-on-gates * this))

(defmethod forward :sig-gate
  [this]
  (core/sig (forward (:gate-a this))))

(defmethod backward :add-gate
  [this back-grad]
  (let [gate-a (:gate-a this)
        gate-b (:gate-b this)]
    (merge-with + (backward gate-a (* 1.0 back-grad))
                (backward gate-b (* 1.0 back-grad)))))

(defmethod backward :mul-gate
  [this back-grad]
  (let [gate-a (:gate-a this)
        gate-b (:gate-b this)
        val-a (forward gate-a)
        val-b (forward gate-b)]
    (merge-with + (backward gate-a (* val-b back-grad))
                (backward gate-b (* val-a back-grad)))))

(defmethod backward :sig-gate
  [this back-grad]
  (let [gate-a (:gate-a this)
        s (forward this)]
    (backward gate-a (* s (- 1 s) back-grad))))

;; f(x,y) = a*x + b*y + c

(defunit a 1.0)
(defunit b 2.0)
(defunit c -3.0)
(defunit x -1.0)
(defunit y 3.0)

(def ax (mul-gate a x))
(def by (mul-gate b y ))
(def axc (addition-gate ax c))
(def axcby (addition-gate axc by))
(def sigaxcby (sig-gate axcby))

;; (forward sigaxcby)
;; (clojure.pprint/pprint (backward sigaxcby 1.0))

;; Plan 1 : DONE :)

;; ---------------- **** -----------------------------
