(ns neurals.core-test
  (:require [clojure.test :refer :all]
            [neurals.core :refer :all]))

(deftest forward-multiply-test
  (testing "Forward-multiply tests"
    (is (= 6 (forward-multiply 2 3)))))

(deftest local-search-better-forward-test
  (testing "Local search should give us some improvement"
    (is (< (forward-multiply -3 -4) 
           (:val (local-search-better-forward -3 -4 30))))))

(deftest x-derivative-test
  (testing "Find the x-derivative going forward"
    (is (= 3.00000000000189 (x-derivative-forward -2 3 0.0001)))))

(deftest y-derivative-test
  (testing "Find the y-derivative going forward"
    (is (= -2.0000000000042206 (y-derivative-forward -2 3 0.0001)))))


(deftest numerical-gradient-forward-test
  (testing "Numerical gradient should be giving us better results."
    (is (< (forward-multiply -3 -4) 
           (:val (numerical-gradient-forward -3 -4))))))

(deftest analytical-gradient-forward-test
  (testing "Analytical gradient should be giving us better results."
    (is (< (forward-multiply -3 -4) 
           (:val (analytical-gradient-forward -3 -4))))))

(deftest better-gradient-forward-test
  (testing "Analytical should be as good as Numerical gradient."
    (is (= (:val (numerical-gradient-forward -3 -4)) 
           (:val (numerical-gradient-forward -3 -4))))))
