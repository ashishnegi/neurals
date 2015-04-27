# neurals

Clojure code examples while learning NeuralNets from `Hacker Guide to Neural Networks` (HGNN).
Link to [HGNN](http://karpathy.github.io/neuralnets/).

## Usage

You can load .clj files in repl.
There are three files (with comments) :

1. `core.clj` : This has the initial local-search and numerical/analytical gradient
              methods. It also uses `defprotocol` for a circuit.
2. `edges.clj` : While learning NN, i also wanted to write a better designed code. 
                 I hit some questions which i asked here: [Designing a Circuit of Gates](http://codereview.stackexchange.com/questions/87536/designing-a-circuit-of-gates-in-clojure-and-doing-forward-and-backpropagation).
							I looked at the circuit from another perpective. It can be thought of as
							a collection of edges, rather than gates. Coding with edges seemed to capture
							the flow in circuit. Coding with gates seemed to capture the state of circuit.
3. `multi.clj` : This is a amalgamation of core.clj's defprotocol and edges.clj's flow-cature thoughts.
	 					 	 Multimethods are used. Maps are used. Circuit becomes a function as a whole. Data flows 
							 in and out. This contains the complete examples of circuit in HGNN.


## License

Copyright © 2015 FIXME

Distributed under the Eclipse Public License either version 2.0.

