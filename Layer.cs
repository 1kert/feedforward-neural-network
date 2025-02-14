using System.Text;

namespace feedforward_neural_network;

public class Layer
{
    readonly double[][] weights;
    readonly double[] biases;
    readonly double[] sums;
    readonly double[] nodeValues;
    
    public Layer(int layerSize, int inputAmount)
    {
        Random rand = new();
        weights = new double[layerSize][];
        biases = new double[layerSize];
        sums = new double[layerSize];
        nodeValues = new double[layerSize];
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
                sb.Append(Math.Round(weights[node][input], 4));
                if (input != weights[node].Length - 1) sb.Append(", ");
            }
            sb.Append("], ");
            if (node == biases.Length - 1) sb.Append(Math.Round(biases[node], 4));
        }
        sb.Append(']');
        return sb.ToString();
    }
    
    static double Activation(double n)
    {
        return n >= 0 ? n : 0.01 * n;
    }
    
    static double ActivationDerivative(double n)
    {
        return n >= 0 ? 1 : 0.01;
    }
    
    public double[] CalculateOutputs(double[] activations)
    {
        int nodes = biases.Length;
        int weightAmount = activations.Length;
        double[] outputs = new double[nodes];
        for(int node = 0; node < nodes; node++)
        {
            outputs[node] = biases[node];
            double[] nodeWeights = weights[node];
            for (int weight = 0; weight < weightAmount; weight++)
            {
                outputs[node] += nodeWeights[weight] * activations[weight];
            }
            sums[node] = outputs[node];
            outputs[node] = Activation(outputs[node]);
        }
        return outputs;
    }
}
