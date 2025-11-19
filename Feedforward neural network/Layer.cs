namespace Feedforward_neural_network;

public class Layer
{
    private readonly double[][] _weights;
    private readonly double[] _biases;
    private readonly double[] _sums;
    private readonly double[] _activations;
    private readonly double[][] _weightGradient;
    private readonly double[] _biasGradient;
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
    
    private static double SigmoidActivationDerivative(double x) => Math.Exp(-x) / Math.Pow(1 + Math.Exp(-x), 2);
    
    private static double LReLu(double x) => x > 0 ? x : 0.01 * x;
    
    private static double LReLuDerivative(double x) => x > 0 ? 1 : 0.01;
    
    private static double InitializeWeight(int inputs, int outputs) => 
        Math.Sqrt(6.0 / (inputs + outputs)) * (2 * Random.Shared.NextDouble() - 1);
    
    public static double Cost(double expected, double actual) => Math.Pow(expected - actual, 2);

    public static double Cost(double[] expected, double[] actual)
    {
        double totalCost = 0;
        for(int i = 0; i < expected.Length; i++) totalCost += Cost(expected[i], actual[i]);
        return totalCost;
    }
    
    public static double CostDerivative(double expected, double actual) => 2 * (expected - actual);

    public void ClearGradients()
    {
        for (int i = 0; i < Length; i++)
        {
            _biasGradient[i] = 0.0;
            for(int j = 0; j < _weights[i].Length; j++) _weightGradient[i][j] = 0.0;
        }
    }

    public double[] CalculateNextChainValues(double[] previousChainValues)
    {
        double[] chainValues = new double[_inputs.Length];

        for (int j = 0; j < _inputs.Length; j++)
        {
            chainValues[j] = 0;
            
            for (int i = 0; i < Length; i++)
                 chainValues[j] += _weights[i][j] * previousChainValues[i];
            
            chainValues[j] *= SigmoidActivationDerivative(_inputs[j]);
        }
        
        return chainValues;
    }
        
    public double[] CalculateOutputChainValues(double[] expected)
    {
        double[] chainValues = new double[Length];
        for (int i = 0; i < Length; i++)
            chainValues[i] = CostDerivative(expected[i], _activations[i]) * SigmoidActivationDerivative(_sums[i]);
        return chainValues;
    }

    public void CalculateGradients(double[] chainValues)
    {
        for (int i = 0; i < Length; i++)
        {
            _biasGradient[i] = chainValues[i];
            for (int j = 0; j < _weights[i].Length; j++)
                _weightGradient[i][j] = _inputs[j] * chainValues[i];
        }
    }

    public void ApplyGradients(double learningRate)
    {
        for (int i = 0; i < Length; i++)
        {
            _biases[i] += learningRate * _biasGradient[i];
            
            for (int j = 0; j < _weights[i].Length; j++)
                _weights[i][j] += learningRate * _weightGradient[i][j];
        }
    }
}