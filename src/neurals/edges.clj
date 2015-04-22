;; Plan 2:
;; But the problem in both the plans are that
;; their is no caching of data that is once calculated 
;; is calculated again.. especially in `backward`.
;; Can i do it the right way ?
;; Is there functional way of doing things that would 
;; generate the data on the go.. and i would start
;; from that data and calculate the `backward` efficiently.
;; Idea :
;; It seems that i should put outputs from the forward pass,
;; and pass it to the backward pass to process upon.
;; This would solve the inefficient backward problem.

;; Another problem :
;; once i get the gradients from backward pass,
;; how am i supposed to tweak the inputs, since they are immutable.
;; This looks like a design flaw..

;; Idea1: Make the circuit and capture the logic not the data.
;; pass the data around in the circuit so that it can work even with
;; new data.

;; Thinking about solving : Rather than modelling gates.. model edges. :)

(ns neurals.edges)

;; every edge knows its input but not its output.
(defmacro definputedge [var-name id]
  `(def ~var-name {:type :input-edge
                   :name (name '~var-name)
                   :id ~id
                   }))

(defn make-random-edge [value gradient input]
  {:type :edge
   :value value
   :gradient gradient
   :name (gensym "edge")
   :id (gensym "id")
   :input input
   })

(declare make-mul-gate)
(defn mul-gate [edge-a edge-b]
  (assoc 
   (make-random-edge 0.0 0.0 (make-mul-gate edge-a edge-b)) 
   :type :mul-gate-edge))

(declare make-add-gate)
(defn add-gate [edge-a edge-b]
  (assoc 
   (make-random-edge 0.0 0.0 (make-add-gate edge-a edge-b)) 
   :type :add-gate-edge))

(declare make-sig-gate)
(defn sig-gate [edge-a]
  (assoc 
   (make-random-edge 0.0 0.0 (make-sig-gate edge-a)) 
   :type :sig-gate-edge))

(defn make-mul-gate [edge-a edge-b]
  {:type :mul-gate
   :edge-a edge-a
   :edge-b edge-b})

(defn make-add-gate [edge-a edge-b]
  {:type :add-gate
   :edge-a edge-a
   :edge-b edge-b})

(defn make-sig-gate [edge-a]
  {:type :sig-gate
   :edge-a edge-a})


(defmulti forward (fn [this _] (:type this)))
(defmulti backward (fn [this _ _] (:type this)))

;; for input-edge which are left-most edges,
;; pass-back the value given in the input.
(defmethod forward :input-edge
  [this input-values]
  (let [id (:id this)]
    {id  (input-values id)}))

(defmethod forward :add-gate-edge
  [this input-values]
  (let [input (:input this)
        id (:id this)
        edge-a (:edge-a input)
        edge-b (:edge-b input)
        id-a (:id edge-a)
        id-b (:id edge-b)
        val-a-map (forward edge-a input-values)
        val-b-map (forward edge-b input-values)]
    (-> (merge-with + val-a-map val-b-map)
        (assoc   id (+ (val-a-map id-a)
                       (val-b-map id-b))))))

(defmethod forward :mul-gate-edge
  [this input-values]
  (let [input (:input this)
        id (:id this)
        edge-a (:edge-a input)
        edge-b (:edge-b input)
        id-a (:id edge-a)
        id-b (:id edge-b)
        val-a-map (forward edge-a input-values)
        val-b-map (forward edge-b input-values)]
    (-> (merge-with + val-a-map val-b-map)
        (assoc   id (* (val-a-map id-a)
                       (val-b-map id-b))))))

(defn sig 
  "Sigmoid function : f(x) = 1/(1 + exp(-x))"
  [x]
  (/ 1 (+ 1 (Math/pow Math/E (- x)))))

(defmethod forward :sig-gate-edge
  [this input-values]
  (let [input (:input this)
        id (:id this)
        edge-a (:edge-a input)
        id-a (:id edge-a)
        val-a-map (forward edge-a input-values)]
    (-> val-a-map
        (assoc   id (sig (val-a-map id-a))))))


(defmethod backward :input-edge
  [this pull values-edges]
  (let [id (:id this)]
    {id pull}))

(defmethod backward :add-gate-edge
  [this pull values-edges]
  (let [id (:id this)
        input (:input this)
        edge-a (:edge-a input)
        edge-b (:edge-b input)
        pull-a (* 1.0 pull)
        pull-b (* 1.0 pull)]
    (-> 
     (merge-with + (backward edge-a pull-a values-edges)
                 (backward edge-b pull-a values-edges))
     (assoc   id pull))))

(defmethod backward :mul-gate-edge
  [this pull values-edges]
  (let [id (:id this)
        input (:input this)
        edge-a (:edge-a input)
        edge-b (:edge-b input)
        id-a (:id edge-a)
        id-b (:id edge-b)
        pull-a (* (values-edges id-b) pull)
        pull-b (* (values-edges id-a) pull)]
    (-> 
     (merge-with + (backward edge-a pull-a values-edges)
                 (backward edge-b pull-a values-edges))
     (assoc   id pull))))

(defmethod backward :sig-gate-edge
  [this pull values-edges]
  (let [id (:id this)
        input (:input this)
        edge-a (:edge-a input)
        val (values-edges id)
        pull-a (* val (- 1 val) pull)]
    (-> 
     (backward edge-a pull-a values-edges)
     (assoc   id pull))))

(definputedge a 0)
(definputedge b 1)
(definputedge c 2)
(definputedge x 3)
(definputedge y 4)

(def ax (mul-gate a x))
(def by (mul-gate b y))
(def axby (add-gate ax by))
(def axbyc (add-gate axby c))
(def sigaxbyc (sig-gate axbyc))

(backward sigaxbyc 1.0 (forward sigaxbyc {0 1.0
                             1 2.0
                             2 -3.0
                             3 -1.0
                             4 3.0
                             }))

;; After doing it in the right way, i think that i could have achieved the 
;; same with defprotocol too.
;; But i like edges better than gates :)
