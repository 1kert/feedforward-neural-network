namespace feedforward_neural_network;

public class DataPoint(double[] inputs, double[] expected)
{
    public double[] Inputs { get; set; } = inputs;
    public double[] Expected { get; set; } = expected;
}
