// Based on http://iamtrask.github.io/2015/07/12/basic-python-network/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using OpenMined.Syft;

public class BasicNeuralNetwork : MonoBehaviour {

	// Use this for initialization
	void Start () {

		System.Diagnostics.Stopwatch stopWatch = new System.Diagnostics.Stopwatch ();

		stopWatch.Start ();
		train ();
		stopWatch.Stop ();
		Debug.LogFormat ("<color=white>BasicNeuralNetwork: ElapsedMilliseconds:</color> {0}", stopWatch.ElapsedMilliseconds);

		stopWatch.Restart ();
		trainWithGPU ();
		stopWatch.Stop ();
		Debug.LogFormat ("<color=cyan>BasicNeuralNetwork.withGPU Elapsed:</color> {0}", stopWatch.ElapsedMilliseconds);

	}
	
	// Update is called once per frame
	void Update () {
		
	}

//	# sigmoid function
//	def nonlin(x,deriv=False):
//		if(deriv==True):
//			return x*(1-x)
//		return 1/(1+np.exp(-x))
	private float sigmoid (float x) {
		return 1 / (1 + Mathf.Exp (-x));
	}

	private float derivative (float x) {
		return x * (1 - x);
	}

	void train () {

//		# input dataset
//		X = np.array([  [0,0,1],
//			[0,1,1],
//			[1,0,1],
//			[1,1,1] ])
		float[][] X = new float[][] {
			new float[] { 0,0,1 },
			new float[] { 0,1,1 },
			new float[] { 1,0,1 },
			new float[] { 1,1,1 }
		};
		Debug.LogFormat ("<color=white>BasicNeuralNetwork: X:</color> [{0}], [{1}], [{2}], [{3}]",
			string.Join(", ", X[0]),
			string.Join(", ", X[1]),
			string.Join(", ", X[2]),
			string.Join(", ", X[3]));

//		# output dataset            
//		y = np.array([[0,0,1,1]]).T
		float[] y = new float[] { 0, 0, 1, 1 };
		Debug.LogFormat ("<color=white>BasicNeuralNetwork: y:</color> {0}", string.Join(", ", y));

//		# seed random numbers to make calculation
//		# deterministic (just a good practice)
//		np.random.seed(1)
		Random.InitState (1);

//		# initialize weights randomly with mean 0
//		syn0 = 2*np.random.random((3,1)) - 1
//		print(syn0)
		float[] syn0 = new float[3];
		for (int i = 0; i < syn0.Length; i++) {
			syn0 [i] = 2 * Random.value - 1;
		}
		//Debug.LogFormat("<color=green>syn0:</color> {0}", string.Join(", ", syn0));

		float[] l1 = new float[4];

//		for iter in range(10000):
		for (int r = 0; r < 10000; r++) {

//			# forward propagation
//			l0 = X
//			l1 = nonlin(np.dot(l0,syn0))
			float[][] l0 = X;
			for (int i = 0; i < 1; i++) {
				for (int j = 0; j < l1.Length; j++) {
					float total = 0;
					for (int k = 0; k < l0 [j].Length; k++) {
						total += l0 [j] [k] * syn0 [k];
						//Debug.LogFormat ("Total: {0}", total);
					}
					l1 [j] = sigmoid (total);
				}
			}
			//Debug.LogFormat ("<color=green>l1:</color> {0}", string.Join(", ", l1));

//			# how much did we miss?
//			l1_error = y - l1
			float[] l1_error = new float[4];
			for (int i = 0; i < l1.Length; i++) {
				l1_error [i] = y [i] - l1 [i];
			}
			//Debug.LogFormat ("<color=green>l1_error:</color> {0}", string.Join(", ", l1_error));

//			# multiply how much we missed by the 
//			# slope of the sigmoid at the values in l1
//			l1_delta = l1_error * nonlin(l1,True)
			float[] l1_delta = new float[4];
			for (int i = 0; i < l1_delta.Length; i++) {
				l1_delta [i] = l1_error [i] * derivative (l1 [i]);
			}
			//Debug.LogFormat ("<color=green>l1_delta:</color> {0}", string.Join(", ", l1_delta));

//			# update weights
//			syn0 += np.dot(l0.T,l1_delta)
			for (int i = 0; i < 1; i++) {
				for (int j = 0; j < syn0.Length; j++) {
					for (int k = 0; k < l0.Length; k++) {
						syn0 [j] += l0 [k] [j] * l1_delta [k];
					}
				}
			}
			//Debug.LogFormat ("<color=green>updated syn0:</color> {0}", string.Join(", ", syn0));
		}

		Debug.LogFormat ("<color=white>BasicNeuralNetwork: Output After Training:</color> {0}", string.Join(", ", l1));
		Debug.LogFormat ("<color=white>BasicNeuralNetwork: Weights After Training:</color> {0}", string.Join(", ", syn0));

	}

	[SerializeField]
	private ComputeShader shader;

	void trainWithGPU () {

		float[] l0_data = new float[] { 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1 };
		int[] l0_shape = new int[] { 3, 3, 3, 3 };
		FloatTensor l0_tensor = new FloatTensor (l0_data, l0_shape, shader);
		Debug.LogFormat ("<color=cyan>BasicNeuralNetwork.withGPU l0_tensor:</color> {0}", string.Join (", ", l0_tensor.data));
		l0_tensor.gpu ();

		float[] l1_data = new float[] { 0, 0, 0, 0 };
		int[] l1_shape = new int[] { 4 };
		FloatTensor l1_tensor = new FloatTensor (l1_data, l1_shape, shader);
		Debug.LogFormat ("<color=cyan>BasicNeuralNetwork.withGPU l1_tensor:</color> {0}", string.Join (", ", l1_tensor.data));
		l1_tensor.gpu ();

		float[] syn0_data = randomWeights (3);
		int[] syn0_shape = new int[] { 3 };
		FloatTensor syn0_tensor = new FloatTensor (syn0_data, syn0_shape, shader);
		syn0_tensor.gpu ();

		l1_tensor.init_sigmoid_matrix_multiply (l0_tensor);
		syn0_tensor.init_add_matrix_multiply (l0_tensor);

		float[] y_data = new float[] { 0, 0, 1, 1 };
		int[] y_shape = new int[] { 4 };

		FloatTensor l1_error_tensor = new FloatTensor (y_data, y_shape, shader);
		l1_error_tensor.gpu ();

		FloatTensor save_error_tensor = new FloatTensor (y_data, y_shape, shader);
		save_error_tensor.gpu ();

		l1_error_tensor.init_weights (save_error_tensor);

		for (int i = 0; i < 10000; i++) {
			l1_tensor.sigmoid_matrix_multiply (l0_tensor, syn0_tensor);

			l1_error_tensor.reset_weights ();

			l1_error_tensor.inline_elementwise_subtract (l1_tensor);

			l1_error_tensor.multiply_derivative (l1_tensor);

			syn0_tensor.add_matrix_multiply (l0_tensor, l1_error_tensor);
		}

		l0_tensor.cpu ();
		l1_tensor.cpu ();
		l1_error_tensor.cpu ();
		save_error_tensor.cpu ();
		syn0_tensor.cpu ();

		Debug.LogFormat ("<color=cyan>BasicNeuralNetwork.withGPU Output After Training:</color> {0}", string.Join (", ", l1_tensor.data));
		Debug.LogFormat ("<color=cyan>BasicNeuralNetwork.withGPU Weights After Training:</color> {0}", string.Join (", ", syn0_tensor.data));
	}

	float[] randomWeights (int length) {
		Random.InitState (1);
		float[] syn0 = new float[length];
		for (int i = 0; i < length; i++) {
			syn0 [i] = 2 * Random.value - 1;
		}
		return syn0;
	}

}
