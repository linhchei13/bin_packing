using System;
using System.IO;
using System.Linq;
using Hexaly.Optimizer;

public class BinPacking : IDisposable
{
    // Number of items
    int nbItems;

    // Capacity of each bin
    int binCapacity;

    // Maximum number of bins
    int nbMaxBins;

    // Minimum number of bins
    int nbMinBins;

    // Weight of each item
    long[] weightsData;

    // Hexaly Optimizer
    HexalyOptimizer optimizer;

    // Decision variables
    HxExpression[] bins;

    // Weight of each bin in the solution
    HxExpression[] binWeights;

    // Whether the bin is used in the solution
    HxExpression[] binsUsed;

    // Objective
    HxExpression totalBinsUsed;

    public BinPacking()
    {
        optimizer = new HexalyOptimizer();
    }

    /* Read instance data */
    void ReadInstance(string fileName)
    {
        using (StreamReader input = new StreamReader(fileName))
        {
            nbItems = int.Parse(input.ReadLine());
            binCapacity = int.Parse(input.ReadLine());

            weightsData = new long[nbItems];
            for (int i = 0; i < nbItems; ++i)
                weightsData[i] = int.Parse(input.ReadLine());

            nbMinBins = (int)Math.Ceiling((double)weightsData.Sum() / binCapacity);
            nbMaxBins = Math.Min(2 * nbMinBins, nbItems);
        }
    }

    public void Dispose()
    {
        if (optimizer != null)
            optimizer.Dispose();
    }

    void Solve(int limit)
    {
        // Declare the optimization model
        HxModel model = optimizer.GetModel();

        bins = new HxExpression[nbMaxBins];
        binWeights = new HxExpression[nbMaxBins];
        binsUsed = new HxExpression[nbMaxBins];

        // Set decisions: bin[k] represents the items in bin k
        for (int k = 0; k < nbMaxBins; ++k)
            bins[k] = model.Set(nbItems);

        // Each item must be in one bin and one bin only
        model.Constraint(model.Partition(bins));

        // Create an array and a function to retrieve the item's weight
        HxExpression weights = model.Array(weightsData);
        HxExpression weightLambda = model.LambdaFunction(i => weights[i]);

        for (int k = 0; k < nbMaxBins; ++k)
        {
            // Weight constraint for each bin
            binWeights[k] = model.Sum(bins[k], weightLambda);
            model.Constraint(binWeights[k] <= binCapacity);

            // Bin k is used if at least one item is in it
            binsUsed[k] = model.Count(bins[k]) > 0;
        }

        // Count the used bins
        totalBinsUsed = model.Sum(binsUsed);

        // Minimize the number of used bins
        model.Minimize(totalBinsUsed);

        model.Close();

        // Parametrize the optimizer
        optimizer.GetParam().SetTimeLimit(limit);

        // Stop the search if the lower threshold is reached
        optimizer.GetParam().SetObjectiveThreshold(0, nbMinBins);

        optimizer.Solve();
    }

    /* Write the solution in a file */
    void WriteSolution(string fileName)
    {
        using (StreamWriter output = new StreamWriter(fileName))
        {
            for (int k = 0; k < nbMaxBins; ++k)
            {
                if (binsUsed[k].GetValue() != 0)
                {
                    output.Write("Bin weight: " + binWeights[k].GetValue() + " | Items: ");
                    HxCollection binCollection = bins[k].GetCollectionValue();
                    for (int i = 0; i < binCollection.Count(); ++i)
                        output.Write(binCollection[i] + " ");
                    output.WriteLine();
                }
            }
        }
    }

    public static void Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.WriteLine("Usage: BinPacking inputFile [solFile] [timeLimit]");
            Environment.Exit(1);
        }
        string instanceFile = args[0];
        string outputFile = args.Length > 1 ? args[1] : null;
        string strTimeLimit = args.Length > 2 ? args[2] : "5";

        using (BinPacking model = new BinPacking())
        {
            model.ReadInstance(instanceFile);
            model.Solve(int.Parse(strTimeLimit));
            if (outputFile != null)
                model.WriteSolution(outputFile);
        }
    }
}
