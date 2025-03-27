using System.Text;

namespace feedforward_neural_network;

public class Layer
{
    private readonly double[][] _weights;
    private readonly double[] _biases;
    private readonly double[] _sums;
    public readonly double[] nodeValues;
    private readonly double[][] _weightGradients;
    private readonly double[] _activations;
    
    public Layer(int layerSize, int inputAmount)
    {
        Random rand = new();
        _activations = new double[layerSize];
        _weights = new double[layerSize][];
        _weightGradients = new double[layerSize][];
        _biases = new double[layerSize];
        _sums = new double[layerSize];
        nodeValues = new double[layerSize];
        double range = 1 / Math.Sqrt(inputAmount);
        
        for (int i = 0; i < layerSize; i++)
        {
            _biases[i] = 0;
            _weights[i] = new double[inputAmount];
            for (int j = 0; j < inputAmount; j++)
                _weights[i][j] = -range + 2 * range * rand.NextDouble();
        }
    }

    public override string ToString()
    {
        StringBuilder sb = new();
        sb.Append('[');
        for(int node = 0; node < _biases.Length; node++)
        {
            sb.Append('[');
            for (int input = 0; input < _weights[node].Length; input++)
            {
                sb.Append(Math.Round(_weights[node][input], 4));
                if (input != _weights[node].Length - 1) sb.Append(", ");
            }
            sb.Append("], ");
            if (node == _biases.Length - 1) sb.Append(Math.Round(_biases[node], 4));
        }
        sb.Append(']');
        return sb.ToString();
    }

    private static double LReLuActivation(double n)
    {
        return n >= 0 ? n : 0.01 * n;
    }

    private static double LReLuActivationDerivative(double n)
    {
        return n >= 0 ? 1 : 0.01;
    }
    
    public double[] CalculateOutputs(double[] inputActivations)
    {
        int nodeCount = _biases.Length;
        int weightCount = inputActivations.Length;
        double[] outputs = new double[nodeCount];
        for(int node = 0; node < nodeCount; node++)
        {
            outputs[node] = _biases[node];
            double[] nodeWeights = _weights[node];
            for (int weight = 0; weight < weightCount; weight++)
            {
                outputs[node] += nodeWeights[weight] * inputActivations[weight];
            }
            _sums[node] = outputs[node];
            outputs[node] = LReLuActivation(outputs[node]);
            _activations[node] = outputs[node];
        }
        return outputs;
    }
    
    public void CalculateOutputNodeValues(double[] expected, double[] outputs)
    {
        if (expected.Length != outputs.Length) throw new ArgumentException("expected and output sizes don't match");
        
        for (int i = 0; i < expected.Length; i++)
        {
            double value = 2 * (expected[i] - outputs[i]) * LReLuActivationDerivative(_sums[i]);
            nodeValues[i] = value;
        }
    }

    public double GetHalfNodeValue(int nodeIndex, int weightIndex)
    {
        return _weights[nodeIndex][weightIndex] * nodeValues[nodeIndex];
    }

    public void CalculateNodeValues(Layer nextLayer)
    {
        int nodeCount = _biases.Length;
        
        for (int node = 0; node < nodeCount; node++)
        {
            double nodeValue = LReLuActivationDerivative(_sums[node]);

            for (int i = 0; i < nextLayer.nodeValues.Length; i++)
            {
                nodeValue += nextLayer.GetHalfNodeValue(i, node);
            }
            
            nodeValues[node] = nodeValue;
        }
    }
}
