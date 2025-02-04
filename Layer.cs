using System.Text;

namespace feedforward_neural_network;

public class Layer
{
    readonly double[][] weights;
    readonly double[] biases;
    
    public Layer(int layerSize, int inputAmount)
    {
        Random rand = new();
        weights = new double[layerSize][];
        biases = new double[layerSize];
        double range = 1 / Math.Sqrt(inputAmount);
        
        for (int i = 0; i < layerSize; i++)
        {
            biases[i] = 0;
            weights[i] = new double[inputAmount];
            for (int j = 0; j < inputAmount; j++) 
                weights[i][j] = -range + 2 * range * rand.NextDouble();
        }
    }

    public override string ToString()
    {
        StringBuilder sb = new();
        sb.Append('[');
        for(int node = 0; node < biases.Length; node++)
        {
            sb.Append('[');
            for (int input = 0; input < weights[node].Length; input++)
            {
                sb.Append(weights[node][input]);
                if (input != weights[node].Length - 1) sb.Append(", ");
            }
            sb.Append("], ");
            if (node == biases.Length - 1) sb.Append(biases[node]);
        }
        sb.Append(']');
        return sb.ToString();
    }
}
