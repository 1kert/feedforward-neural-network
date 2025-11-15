namespace Feedforward_neural_network;

public class Layer
{
    private readonly double[][] _weights;
    private readonly double[] _biases;
    private readonly double[] _sums;
    private readonly double[] _activations;
    private readonly double[][] _weightGradient;
    private readonly double[] _biasGradient;
    private readonly double[] _chainValues;
    private double[] _inputs = [];
    
    public int Length => _weights.Length;

    public Layer(int inputs, int outputs)
    {
        _biases = new double[outputs];
        _biasGradient = new double[outputs];
        _sums = new double[outputs];
        _activations = new double[outputs];
        _weights = new double[outputs][];
        _weightGradient = new double[outputs][];
        _chainValues = new double[outputs];
        for (int output = 0; output < outputs; output++)
        {
            _biases[output] = 0;
            _sums[output] = 0;
            _activations[output] = 0;
            _weights[output] = new double[inputs];
            _weightGradient[output] = new double[inputs];
            for (int input = 0; input < inputs; input++) _weights[output][input] = InitializeWeight(inputs, outputs);
        }
    }

    public double[] Calculate(double[] inputs)
    {
        // if (inputs.Length != _weights[0].Length) throw new ArgumentException("Input lengths don't match");
        
        _inputs = inputs;
        double[] sums = new double[Length];
        for (int i = 0; i < Length; i++)
        {
            sums[i] = _biases[i];
            for (int j = 0; j < _weights[i].Length; j++)
                sums[i] += _weights[i][j] * inputs[j];
            
            _sums[i] = sums[i];
            sums[i] = SigmoidActivation(sums[i]);
            _activations[i] = sums[i];
        }
        return sums;
    }

    private static double SigmoidActivation(double x) => 1 / (1 + Math.Exp(-x));
    
    private static double SigmoidActivationDerivative(double x) => Math.Exp(-x) / (1 + Math.Exp(-x));
    
    private static double InitializeWeight(int inputs, int outputs) => 
        Math.Sqrt(6 / (inputs + outputs)) * (2 * Random.Shared.NextDouble() - 1);
    
    public static double Cost(double expected, double actual) => Math.Pow(expected - actual, 2);

    public static double Cost(double[] expected, double[] actual)
    {
        double totalCost = 0;
        for(int i = 0; i< expected.Length; i++) totalCost += Cost(expected[i], actual[i]);
        return totalCost;
    }
    
    public static double CostDerivative(double expected, double actual) => 2 * (expected - actual);

    public void ClearGradients()
    {
        for (int i = 0; i < Length; i++)
        {
            _biasGradient[i] = 0.0;
            for(int j = 0; j < _weights[i].Length; j++) _weights[i][j] = 0.0;
        }
    }

    public double[] CalculateOutputChainValues(double[] expected)
    {
        for (int i = 0; i < Length; i++)
            _chainValues[i] = CostDerivative(expected[i], _activations[i]) * SigmoidActivationDerivative(_sums[i]);
        return _chainValues;
    }

    public void CalculateGradients()
    {
        for (int i = 0; i < Length; i++)
        {
            _biasGradient[i] = _chainValues[i];
            for (int j = 0; j < _weights[i].Length; j++)
                _weightGradient[i][j] = _inputs[j] * _chainValues[i];
        }
    }
}