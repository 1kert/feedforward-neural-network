namespace Feedforward_neural_network;

class Program
{
    static Datapoint[][] DivideToBatches(Datapoint[] dataPoints, int batchSize)
    {
        var copy = new Datapoint[dataPoints.Length];
        Array.Copy(dataPoints, copy, dataPoints.Length);
        Random.Shared.Shuffle(copy);
        var batch = new Datapoint[batchSize];
        List<Datapoint[]> batches = [];
        for (int i = 0; i < dataPoints.Length; i++)
        {
            if (i != 0 && i % batchSize == 0)
            {
                batches.Add(batch);
                batch = new Datapoint[batchSize];
            }
            
            batch[i % batchSize] = copy[i];
        }

        return batches.ToArray();
    }
    
    static void Main()
    {
        const double learningRate = 0.04;
        const int batchSize = 4;
        
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

        int count = 0;
        while (true)
        {
            count++;
            var batches = DivideToBatches(dataPoints, batchSize);

            foreach (var batch in batches)
                network.Learn(batch, learningRate);

            var cost = network.Cost(dataPoints);
            if (count == 1000)
            {
                count = 0;
                Console.WriteLine($"cost: {cost}");
            }
            if (cost <= 0.001) break;
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