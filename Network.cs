namespace feedforward_neural_network;

public class Network
{
    readonly Layer[] layers;

    public Network(params int[] layerSizes)
    {
        layers = new Layer[layerSizes.Length - 1];
        for (int i = 1; i < layerSizes.Length; i++)
            layers[i - 1] = new(layerSizes[i], layerSizes[i - 1]);
    }
    
    public void PrintLayers()
    {
        foreach(Layer layer in layers)
        {
            Console.WriteLine(layer);
        }
    }
    
    public double[] CalculateOutputs(double[] inputs)
    {
        double[] output = inputs;

        foreach(Layer layer in layers)
            output = layer.CalculateOutputs(output);
        
        return output;
    }
    
    static double Cost(double actual, double expected)
    {
        return Math.Pow(actual - expected, 2);
    }
    
    static double Cost(double[] actual, double[] expected)
    {
        double cost = 0;
        for(int i = 0; i < actual.Length; i++) cost += Cost(actual[i], expected[i]);
        return cost;
    }
    
    public double CalculateCost(DataPoint dataPoints)
    {
        double[] outputs = CalculateOutputs(dataPoints.Inputs);
        return Cost(outputs, dataPoints.Expected);
    }
}
