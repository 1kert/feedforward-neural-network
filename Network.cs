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
}
