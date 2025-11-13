namespace Feedforward_neural_network;

public class Layer
{
    private readonly double[][] _weights;
    private readonly double[] _biases;
    private readonly double[] _sums;
    private readonly double[] _activations;
    
    public int Length => _weights.Length;

    public Layer(int inputs, int outputs)
    {
        _biases = new double[outputs];
        _sums = new double[outputs];
        _activations = new double[outputs];
        _weights = new double[outputs][];
        for (int output = 0; output < outputs; output++)
        {
            _biases[output] = 0;
            _sums[output] = 0;
            _activations[output] = 0;
            _weights[output] = new double[inputs];
            for (int input = 0; input < inputs; input++) _weights[output][input] = InitializeWeight(inputs, outputs);
        }
    }

    public double[] Evaluate(double[] inputs)
    {
        // if (inputs.Length != _weights[0].Length) throw new ArgumentException("Input lengths don't match");
        
        double[] sums = new double[Length];
        for (int i = 0; i < Length; i++)
        {
            sums[i] = _biases[i];
            for (int j = 0; j < _weights[i].Length; j++)
                sums[i] += _weights[i][j] * inputs[j];
            sums[i] = SigmoidActivation(sums[i]);
        }
        return sums;
    }

    private static double SigmoidActivation(double x) => 1 / (1 + Math.Exp(-x));
    
    private static double InitializeWeight(int inputs, int outputs) => 
        Math.Sqrt(6 / (inputs + outputs)) * (2 * Random.Shared.NextDouble() - 1);
}