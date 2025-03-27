namespace feedforward_neural_network;

public class Network
{
    private readonly Layer[] _layers;

    public Network(params int[] layerSizes)
    {
        _layers = new Layer[layerSizes.Length - 1];
        for (int i = 1; i < layerSizes.Length; i++)
            _layers[i - 1] = new Layer(layerSizes[i], layerSizes[i - 1]);
    }
    
    public void PrintLayers()
    {
        foreach(Layer layer in _layers)
        {
            Console.WriteLine(layer);
        }
    }
    
    public double[] CalculateOutputs(double[] inputs)
    {
        double[] output = inputs;

        foreach(Layer layer in _layers)
            output = layer.CalculateOutputs(output);
        
        return output;
    }
    
    private static double Cost(double actual, double expected)
    {
        return Math.Pow(actual - expected, 2);
    }
    
    private static double Cost(double[] actual, double[] expected)
    {
        double cost = 0;
        for(int i = 0; i < actual.Length; i++) cost += Cost(actual[i], expected[i]);
        return cost;
    }
    
    public double CalculateCost(DataPoint dataPoint)
    {
        double[] outputs = CalculateOutputs(dataPoint.Inputs);
        return Cost(outputs, dataPoint.Expected);
    }
    
    public void CalculateGradients(DataPoint dataPoint)
    {
        double[] outputs = CalculateOutputs(dataPoint.Inputs);
        double[] expected  = dataPoint.Expected;
        int layerCount = _layers.Length;
        
        Layer outputLayer = _layers[^1];
        outputLayer.CalculateOutputNodeValues(expected, outputs);
        for (int layer = _layers.Length - 1; layer >= 0; layer--)
        {
            
        }
    }
}
