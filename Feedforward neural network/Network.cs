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

    public double[] Calculate(double[] inputs)
    {
        double[] outputs = new double[inputs.Length];
        Array.Copy(inputs, outputs, inputs.Length);
        foreach (Layer layer in _layers)
        {
            outputs = layer.Calculate(outputs);
        }
        
        return outputs;
    }

    public void Learn(Datapoint[] dataPoints, double learningRate)
    {
        // todo: impl batch learning
        foreach (var layer in _layers) layer.ClearGradients();
        
        foreach (var datapoint in dataPoints)
        {
            Calculate(datapoint.Inputs);
            double[] chainValues = _layers[^1].CalculateOutputChainValues(datapoint.Expected);
            for (int i = _layers.Length - 1; i >= 0; i--)
            {
                if (i != _layers.Length - 1)
                    chainValues = _layers[i + 1].CalculateNextChainValues(chainValues);
                _layers[i].CalculateGradients(chainValues);
            }
        }
        
        foreach (var layer in _layers) layer.ApplyGradients(learningRate, dataPoints.Length);
    }

    public double Cost(Datapoint datapoint)
    {
        double[] outputs = Calculate(datapoint.Inputs);
        double cost = Layer.Cost(datapoint.Expected, outputs);
        return cost;
    }

    public double Cost(Datapoint[] dataPoints) => dataPoints.Sum(Cost);
}