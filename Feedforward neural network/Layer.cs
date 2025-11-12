namespace Feedforward_neural_network;

public class Layer
{
    private readonly double[][] _weights;
    private readonly double[] _biases;
    private readonly double[] _sums;
    private readonly double[] _activations;

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
            for (int input = 0; input < inputs; input++) _weights[output][input] = InitializeWeight();
        }
    }
    
    private static double InitializeWeight() => Random.Shared.NextDouble(); // todo: better initialization
}