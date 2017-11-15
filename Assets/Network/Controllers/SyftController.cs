using System.Collections;
using System.Collections.Generic;
using System.Linq;

using UnityEngine;

using OpenMined.Syft;
using OpenMined.Network.Utils;


namespace OpenMined.Network.Controllers {


public class SyftController {

	[SerializeField]
	private ComputeShader shader;

	private List<FloatTensor> tensors;

	public SyftController(ComputeShader _shader)
	{
		shader = _shader;

		tensors = new List<FloatTensor>();

		test_inline_elementwise_subtract ();

	}

	public string processMessage(string json_message) {

		Debug.LogFormat("<color=green>SyftController.processMessage {0}</color>", json_message);
	
		Command cmd = JsonUtility.FromJson<Command>(json_message);

//		if (cmd.objectType == "FloatTensor") {
//			if (cmd.functionCall == "FloatTensor") {
//				FloatTensor x = new FloatTensor (fdata, fshape, shader);
//				
//				tensors.Add (x);
//			}
//		}
		Debug.Log("Object Type:" + (cmd.objectType));
		return "WE DID IT!!!";




//		var splittedStrings = message.Split(' ');
//
//		if (splittedStrings [0] == "0") { // command to create a new object of some type
//
//			Debug.Log("<color=green>SyftController.processMessage: Create a tensor object</color>");
//
//			float[] fdata = new float[splittedStrings.Length - 1];
//			for (int i = 0; i < splittedStrings.Length - 1; i++) {
//				fdata [i] = float.Parse (splittedStrings [i + 1]);
//			}
//
//			int[] fshape = new int[1];
//			fshape [0] = splittedStrings.Length - 1;
//
//			FloatTensor x = new FloatTensor (fdata, fshape, shader);
//
//			tensors.Add (x);
//
//			string created = string.Join(",", x.data);
//			Debug.LogFormat("<color=green>SyftController.processMessage: FloatTensor created: {0}</color>", created);
//
//		} else if (splittedStrings [0] == "1") { // command to do something with a Tensor object
//
//			Debug.Log("<color=green>SyftController.processMessage: Execute a tensor object command</color>");
//
//			int tensor_index = int.Parse (splittedStrings [1]);
//
//			FloatTensor tensor = tensors [tensor_index];
//
//			int message_offset = 2;
//
//			string command = splittedStrings [message_offset];
//			Debug.LogFormat("<color=green>SyftController.processMessage command: {0}</color>", command);
//
//			if (command == "0") { // command to call scalar_mult
//				float factor = (float)int.Parse (splittedStrings [message_offset + 1]);
//				Debug.LogFormat ("<color=green>SyftController.processMessage factor: {0}</color>", factor);
//
//				string before = string.Join (",", tensor.data);
//
//				tensor.scalar_mult (factor);
//
//				string after = string.Join (",", tensor.data);
//
//				Debug.LogFormat ("<color=green>SyftController.processMessage answer: {0} * {1} = {2}</color>", before, factor, after);
//
//			} else if (command == "1") { // command to call inline_elementwise_subtract
//				int other_tensor_index = int.Parse (splittedStrings [message_offset + 1]);
//				Debug.LogFormat ("<color=green>SyftController.processMessage other_tensor_index: {0}</color>", other_tensor_index);
//
//				FloatTensor other_tensor = tensors [other_tensor_index];
//
//				string before = string.Join (",", tensor.data);
//
//				string other_tensor_data = string.Join (",", other_tensor.data);
//
//				tensor.inline_elementwise_subtract (other_tensor);
//
//				string after = string.Join (",", tensor.data);
//
//				Debug.LogFormat ("<color=green>SyftController.processMessage answer: {0} - {1} = {2}</color>", before, other_tensor_data, after);
//
//			}
//		}

	}

	void test_inline_elementwise_subtract () {

		float[] xdata = new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
		int[] xshape = new int[] { 3, 3, 3, 3 };
		FloatTensor xtensor = new FloatTensor (xdata, xshape, shader);
		xtensor.gpu ();

		//float[] ydata = new float[] { 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1 };
		float[] ydata = new float[] { 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
		int[] yshape = new int[] { 3, 3, 3, 3 };
		FloatTensor ytensor = new FloatTensor (ydata, yshape, shader);
		ytensor.gpu ();

		xtensor.inline_elementwise_subtract (ytensor);

		xtensor.cpu ();
		ytensor.cpu ();

		Debug.LogFormat ("<color=green>SyftController.test_inline_elementwise_subtract: {0}</color>", string.Join (",", xtensor.data));
	}
}

}
