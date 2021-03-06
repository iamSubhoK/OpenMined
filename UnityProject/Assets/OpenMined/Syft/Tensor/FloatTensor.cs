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
			

			switch (msgObj.functionCall)
			{
			case "abs_":
				{
					// calls the function on our tensor object
					this.Abs_();
					// returns the function call name with the OK status    
					return this.Id + "";
				}
			case "add_":
				{
					this.Add_((float)msgObj.tensorIndexParams[0]); 
					return msgObj.functionCall + ": OK";
				}
			case "add":
				{
					FloatTensor tensor_1 = ctrl.getTensor(msgObj.tensorIndexParams [0]);
					FloatTensor output = tensor_1.Add (tensor_1);
					return ctrl.addTensor (output).ToString ();
				}

			case "addmm_":
				{
					FloatTensor tensor_1 = ctrl.getTensor(msgObj.tensorIndexParams [0]);
					FloatTensor tensor_2 = ctrl.getTensor(msgObj.tensorIndexParams [1]);
					this.AddMatrixMultiply (tensor_1, tensor_2);
					return msgObj.functionCall + ": OK";
				}
			case "ceil":
				{
					this.Ceil (); 
					return msgObj.functionCall + ": OK";
				}
			case "cpu":
				{
					this.Cpu(); 
					return msgObj.functionCall + ": OK";
				}
			case "gpu":
				{
					if (this.Gpu())
					{
						return msgObj.functionCall + ": OK : Moved data to GPU.";
					}
					else
					{
						return msgObj.functionCall + ": FAILED : Did not move data.";
					}
				}

			case "mul":
				{
					FloatTensor tensor_1 = ctrl.getTensor(msgObj.tensorIndexParams [0]);
					this.MulElementwise (tensor_1);
					return msgObj.functionCall + ": OK";
				}
			case "mul_scalar":
				{
					this.MulScalar((float)msgObj.tensorIndexParams[0]);
					return msgObj.functionCall + ": OK";
				}
			case "neg":
				{
					this.Neg ();
					return msgObj.functionCall + ": OK";
				}

			case "print":
				{
					bool dataOriginallyOnGpu = dataOnGpu;
					if (dataOnGpu)
					{
						this.Cpu();
					}

					string data = string.Join (", ", this.Data);
					Debug.LogFormat ("<color=cyan>print:</color> {0}", data);

					if (dataOriginallyOnGpu)
					{
						this.Gpu();
					}
					
					return data;

				}
			case "sub_":
				{
					FloatTensor tensor_1 = ctrl.getTensor(msgObj.tensorIndexParams [0]);
					this.ElementwiseSubtract (tensor_1);
					return msgObj.functionCall + ": OK";
				}
			case "zero_":
				{
					this.Zero_ (); 
					return msgObj.functionCall + ": OK";
				}
			default: break;
			}
			return "SyftController.processMessage: Command not found.";
		}

        public string Print()
        {

	        bool dataOriginallyOnGpu = dataOnGpu;
            if (dataOnGpu)
            {
	            Cpu();
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

	        if (dataOriginallyOnGpu)
	        {
		        Gpu();

	        }
	        return print;
        }
    }
}