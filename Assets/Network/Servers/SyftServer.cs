using UnityEngine;

using OpenMined.Network.Controllers;
using OpenMined.Syft;


namespace OpenMined.Network.Servers
{
public class SyftServer : MonoBehaviour
{
    public bool Connected;
    private NetMqPublisher _netMqPublisher;
    private string _response;

	private SyftController controller;

	[SerializeField]
	private ComputeShader shader;

    private void Start()
    {
        _netMqPublisher = new NetMqPublisher(HandleMessage);
        _netMqPublisher.Start();

		controller = new SyftController (shader);
    }

    private void Update()
    {
        //TODO: Is this code below really required?
        //Connected = _netMqPublisher.Connected;
    }

    private string HandleMessage(string message)
    {
        // Not on main thread
		return controller.processMessage (message);;
    }

    private void OnDestroy()
    {
        _netMqPublisher.Stop();
    }
}
}