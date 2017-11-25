using System;
using UnityEngine;
using OpenMined.Network.Utils;
using OpenMined.Network.Controllers;

namespace OpenMined.Syft.Tensor
{
    public partial class FloatTensor
    {
        // Should we put a check incase this variable overflows?
        private static volatile int nCreated = 0;

        private float[] data;
        private long[] strides;
        private int[] shape;
        private int size;
        
        private int id;
        
        private long GetIndex(params int[] indices)
        {
            long offset = 0;
            for (int i = 0; i < indices.Length; ++i)
            {
                if(indices[i] >= shape[i] || indices[i] < 0)
                    throw new IndexOutOfRangeException();
                offset += indices[i] * strides[i];
            }
            return offset;
        }

        public float[] Data
        {
            get { return data; }            
        }

        public int[] Shape
        {
            get { return shape; }
        }

        public int Size
        {
            get { return size; }
        }

        public int Id
        {
            get { return id; }
            
            set { id = value; }
        }

        public static int CreatedObjectCount
        {
            get { return nCreated; }
        }

        public FloatTensor(int[] _shape, bool _initOnGpu = false) {

            this.size = 1;
            this.shape = (int[])_shape.Clone();
            this.strides = new long[_shape.Length];

            for (var i = _shape.Length - 1; i >= 0; --i)
            {
                this.strides[i] = this.size;
                this.size *= _shape[i];
            }

            if (_initOnGpu)
            {
                this.dataOnGpu = true;
                this.dataBuffer = new ComputeBuffer(this.size, sizeof(float));
                this.shapeBuffer = new ComputeBuffer(this.shape.Length, sizeof(int));
            }
            else
            {
                this.data = new float[this.size];
            }
            
            this.id = System.Threading.Interlocked.Increment(ref nCreated);
        }

        public FloatTensor(float[] _data, int[] _shape, bool _initOnGpu = false)
        {
            //TODO: Can contigous allocation might be a problem?

            if (_shape == null || _shape.Length == 0) {
                throw new InvalidOperationException("Tensor shape can't be an empty array.");
            }
            
            this.size = _data.Length;
            this.shape = (int[])_shape.Clone();
            this.strides = new long[_shape.Length];

            long acc = 1;
            for (var i = _shape.Length - 1; i >= 0; --i)
            {
                this.strides[i] = acc;
                acc *= _shape[i];
            }

            if (acc != this.size)
                throw new FormatException("Tensor shape and data do not match.");

            if (_initOnGpu)
            {
                this.dataOnGpu = true;
                
                this.dataBuffer = new ComputeBuffer(this.size, sizeof(float));
                this.dataBuffer.SetData(_data);	
                
                this.shapeBuffer = new ComputeBuffer(this.shape.Length, sizeof(int));
                this.shapeBuffer.SetData(this.shape);
            }
            else
            {
                this.data = (float[])_data.Clone();
            }

            // IDEs might show a warning, but ref and volatile seems to be working with Interlocked API.
            this.id = System.Threading.Interlocked.Increment(ref nCreated); 
        }
        
        public float this[params int[] indices]
        {
            get
            {
                return Data[GetIndex(indices)];
            }
            set
            {
                Data[GetIndex(indices)] = value;
            }
        }

		public string processMessage(Command msgObj, SyftController ctrl) {
			bool success = true;

			FloatTensor tensor = ctrl.getTensor(msgObj.objectIndex);

			if (msgObj.functionCall == "init_add_matrix_multiply") {

				FloatTensor tensor_1 = ctrl.getTensor(msgObj.tensorIndexParams [0]);
				tensor.ElementwiseMultiplication (tensor_1);

			} else if (msgObj.functionCall == "inline_elementwise_subtract") {

				FloatTensor tensor_1 = ctrl.getTensor(msgObj.tensorIndexParams [0]);
				tensor.ElementwiseSubtract (tensor_1);

			} else if (msgObj.functionCall == "multiply_derivative") {

				FloatTensor tensor_1 = ctrl.getTensor(msgObj.tensorIndexParams [0]);
				tensor.MultiplyDerivative (tensor_1);

			} else if (msgObj.functionCall == "add_matrix_multiply") {

				FloatTensor tensor_1 = ctrl.getTensor(msgObj.tensorIndexParams [0]);
				FloatTensor tensor_2 = ctrl.getTensor(msgObj.tensorIndexParams [1]);

				tensor.AddMatrixMultiply (tensor_1, tensor_2);

			} else if (msgObj.functionCall == "print") {
				
				return tensor.Print ();

			} else if (msgObj.functionCall == "gpu") {
				
				tensor.Gpu ();

			} else if (msgObj.functionCall == "cpu") {
				
				tensor.Cpu ();

			}else if (msgObj.functionCall == "abs") {
				
				// calls the function on our tensor object
				tensor.Abs ();

			}
			else if (msgObj.functionCall == "neg") {
				
				tensor.Neg ();

			}
			else if (msgObj.functionCall == "add") {
				
				FloatTensor tensor_1 = ctrl.getTensor(msgObj.tensorIndexParams [0]);

				FloatTensor output = tensor_1.Add (tensor_1);
				string id = ctrl.addTensor(output).ToString();
				return id;

			}
			else if (msgObj.functionCall == "add_"){
				
				tensor.Add_((float)msgObj.tensorIndexParams[0]);

			}

			else if (msgObj.functionCall == "scalar_multiply")
			{
				
				//get the scalar, cast it and multiply
				tensor.ScalarMultiplication((float)msgObj.tensorIndexParams[0]);

			}
			else if (msgObj.functionCall == "zero_") {
				
				tensor.Zero_ ();

			}
			else
			{
				success = false;
			}

			if (success) {
				
				return msgObj.functionCall + ": OK";

			} else {
				
				return "SyftController.processMessage: Command not found.";

			}
		}


        public string Print()
        {
            if (dataOnGpu)
            {
                CopyGpuToCpu();
            }

            string print = "";

            if (shape.Length > 3)
                print += "Only printing the last 3 dimesnions\n";
            int d3 = 1;
            if (shape.Length > 2)
                d3 = shape[shape.Length - 3];
            int d2 = 1;
            if (shape.Length > 1)
                d2 = shape[shape.Length-2];
            int d1 = shape[shape.Length-1];

            for (int k = 0; k < d3; k++)
            {
                for (int j = 0; j < d2; j++)
                {
                    for (int i = 0; i < d1; i++)
                    {
                        float f = data[i + j * d2 + k * d1 * d2 ];
                        print += f.ToString() + ",\t";
                    }
                    print += "\n";
                }
                print += "\n";
            }
            return print;
        }
    }
}