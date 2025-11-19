namespace Feedforward_neural_network;

class Program
{
    static void Main(string[] args)
    {
        Network network = new(1, 4, 2);
        Datapoint[] dataPoints =
        [
            new([0.6], [1, 0]),
            new([0.7], [1, 0]),
            new([0.8], [1, 0]),
            new([0.9], [1, 0]),
            new([0.54], [1, 0]),
            new([0.2], [0, 1]),
            new([0], [0, 1]),
            new([0.48], [0, 1]),
            new([0.10], [0, 1])
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
            var outputs = network.Calculate(datapoint.Inputs).Select(x => Math.Round(x, 2));
            Console.WriteLine($"[{string.Join(", ", datapoint.Inputs)}]: [{string.Join(", ", outputs)}]");
        }
    }
}