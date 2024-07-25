import java.util.*;
import java.io.*;
import com.hexaly.optimizer.*;

public class BinPacking {
    // Number of items
    private int nbItems;

    // Capacity of each bin
    private int binCapacity;

    // Maximum number of bins
    private int nbMaxBins;

    // Minimum number of bins
    private int nbMinBins;

    // Weight of each item
    private long[] weightsData;

    // Hexaly Optimizer
    private final HexalyOptimizer optimizer;

    // Decision variables
    private HxExpression[] bins;

    // Weight of each bin in the solution
    private HxExpression[] binWeights;

    // Whether the bin is used in the solution
    private HxExpression[] binsUsed;

    // Objective
    private HxExpression totalBinsUsed;

    private BinPacking(HexalyOptimizer optimizer) {
        this.optimizer = optimizer;
    }

    /* Read instance data */
    private void readInstance(String fileName) throws IOException {
        try (Scanner input = new Scanner(new File(fileName))) {
            nbItems = input.nextInt();
            binCapacity = input.nextInt();

            weightsData = new long[nbItems];
            for (int i = 0; i < nbItems; ++i) {
                weightsData[i] = input.nextInt();
            }

            long sumWeights = 0;
            for (int i = 0; i < nbItems; ++i) {
                sumWeights += weightsData[i];
            }

            nbMinBins = (int) Math.ceil((double) sumWeights / binCapacity);
            nbMaxBins = Math.min(2 * nbMinBins, nbItems);
        }
    }

    private void solve(int limit) {
        // Declare the optimization model
        HxModel model = optimizer.getModel();

        bins = new HxExpression[nbMaxBins];
        binWeights = new HxExpression[nbMaxBins];
        binsUsed = new HxExpression[nbMaxBins];

        // Set decisions: bins[k] represents the items in bin k
        for (int k = 0; k < nbMaxBins; ++k) {
            bins[k] = model.setVar(nbItems);
        }

        // Each item must be in one bin and one bin only
        model.constraint(model.partition(bins));

        // Create an array and a lambda function to retrieve the item's weight
        HxExpression weights = model.array(weightsData);
        HxExpression weightLambda = model.lambdaFunction(i -> model.at(weights, i));

        for (int k = 0; k < nbMaxBins; ++k) {
            // Weight constraint for each bin
            binWeights[k] = model.sum(bins[k], weightLambda);
            model.constraint(model.leq(binWeights[k], binCapacity));

            // Bin k is used if at least one item is in it
            binsUsed[k] = model.gt(model.count(bins[k]), 0);
        }

        // Count the used bins
        totalBinsUsed = model.sum(binsUsed);

        // Minimize the number of used bins
        model.minimize(totalBinsUsed);
        model.close();

        // Parametrize the optimizer
        optimizer.getParam().setTimeLimit(limit);

        // Stop the search if the lower threshold is reached
        optimizer.getParam().setObjectiveThreshold(0, nbMinBins);

        optimizer.solve();
    }

    /* Write the solution in a file */
    private void writeSolution(String fileName) throws IOException {
        try (PrintWriter output = new PrintWriter(fileName)) {
            for (int k = 0; k < nbMaxBins; ++k) {
                if (binsUsed[k].getValue() != 0) {
                    output.print("Bin weight: " + binWeights[k].getValue() + " | Items: ");
                    HxCollection binCollection = bins[k].getCollectionValue();
                    for (int i = 0; i < binCollection.count(); ++i) {
                        output.print(binCollection.get(i) + " ");
                    }
                    output.println();
                }
            }
        }
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java BinPacking inputFile [outputFile] [timeLimit]");
            System.exit(1);
        }

        String instanceFile = args[0];
        String outputFile = args.length > 1 ? args[1] : null;
        String strTimeLimit = args.length > 2 ? args[2] : "5";

        try (HexalyOptimizer optimizer = new HexalyOptimizer()) {
            BinPacking model = new BinPacking(optimizer);
            model.readInstance(instanceFile);
            model.solve(Integer.parseInt(strTimeLimit));
            if (outputFile != null) {
                model.writeSolution(outputFile);
            }
        } catch (Exception ex) {
            System.err.println(ex);
            ex.printStackTrace();
            System.exit(1);
        }
    }
}
