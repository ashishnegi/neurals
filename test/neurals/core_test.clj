(ns neurals.core-test
  (:require [clojure.test :refer :all]
            [neurals.core :refer :all]))

(deftest a-test
  (testing "FIXME, I fail."
    (is (= 0 1))))

(deftest x-derivative-test
  (testing "Find the x-derivative going forward"
    (is (= 3.00000000000189 (x-derivative-forward -2 3 0.0001)))))

(deftest y-derivative-test
  (testing "Find the y-derivative going forward"
    (is (= -2.0000000000042206 (y-derivative-forward -2 3 0.0001)))))

