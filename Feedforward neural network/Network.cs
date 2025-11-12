namespace Feedforward_neural_network;

public class Network
{
    private readonly Layer[] _layers;

    public Network(params int[] layers)
    {
        _layers = new Layer[layers.Length - 1];
        for (int i = 1; i < layers.Length; i++)
            _layers[i - 1] = new Layer(layers[i - 1], layers[i]);
    }

    public double[] Evaluate(double[] inputs)
    {
        if (inputs.Length != _layers[0].Length) throw new ArgumentException("Input lengths don't match");

        double[] outputs = inputs;
        foreach (Layer layer in _layers)
        {
            outputs = layer.Evaluate(inputs);
        }
        
        return outputs;
    }
}