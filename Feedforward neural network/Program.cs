namespace Feedforward_neural_network;

class Program
{
    static void Main(string[] args)
    {
        Network network = new(2, 2, 2);
        Datapoint datapoint = new([0.2, 0.9], [1, 0]);

        for (int i = 0; i < 100_000; i++)
        {
            double[] output = network.Calculate(datapoint.Inputs);
            Console.WriteLine($"[{string.Join(", ", output)}]");
            network.Learn(datapoint, 0.2);
        }
    }
}