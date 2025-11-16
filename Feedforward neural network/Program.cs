namespace Feedforward_neural_network;

class Program
{
    static void Main(string[] args)
    {
        Network network = new(2, 8, 8, 4, 2); // todo: fix
        Datapoint[] dataPoints =
        [
            new([0.2, 0.3], [1, 0]),
            new([0.3, 0.4], [0.5, 0.5]),
            new([0.4, 0.5], [0, 0])
        ];
        
        const double learningRate = 0.02;

        for (int i = 0; i < 100_000; i++)
        {
            foreach (Datapoint datapoint in dataPoints)
            {
                network.Learn(datapoint, learningRate);
            }

            // double[] output = network.Calculate(datapoint.Inputs);
            // network.Learn(datapoint, 0.2);
        }
        
        foreach (Datapoint datapoint in dataPoints)
        {
            Console.WriteLine($"[{string.Join(", ", network.Calculate(datapoint.Inputs))}]");
        }
    }
}