namespace Feedforward_neural_network;

class Program
{
    static void Main()
    {
        Network network = new(1, 4, 4, 4, 2);
        Datapoint[] dataPoints =
        [
            new([0.6], [1, 0]),
            new([0.7], [1, 0]),
            new([0.8], [1, 0]),
            new([0.9], [1, 0]),
            new([0.54], [1, 0]),
            new([0.5], [1, 0]),
            new([0.2], [0, 1]),
            new([0], [0, 1]),
            new([0.48], [0, 1]),
            new([0.10], [0, 1])
        ];
        
        const double learningRate = 0.02;

        while (true)
        {
            foreach (Datapoint datapoint in dataPoints)
            {
                network.Learn(datapoint, learningRate);
            }
            
            if (network.Cost(dataPoints) <= 0.001) break;
        }
        
        foreach (Datapoint datapoint in dataPoints)
        {
            var outputs = network.Calculate(datapoint.Inputs).Select(x => Math.Round(x, 2));
            Console.WriteLine($"[{string.Join(", ", datapoint.Inputs)}]: [{string.Join(", ", outputs)}]");
        }

        while (true)
        {
            string input = Console.ReadLine()!;
            if (double.TryParse(input, out double num))
            {
                double[] outputs = network.Calculate([num]);
                string ans = outputs[0] > outputs[1] ? "Yes" : "No";
                Console.WriteLine(ans);
            }
        }
    }
}