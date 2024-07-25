#include "optimizer/hexalyoptimizer.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

using namespace hexaly;
using namespace std;

class BinPacking {
private:
    // Number of items
    int nbItems;

    // Capacity of each bin
    int binCapacity;

    // Maximum number of bins
    int nbMaxBins;

    // Minimum number of bins
    int nbMinBins;

    // Weight of each item
    std::vector<hxint> weightsData;

    // Hexaly Optimizer
    HexalyOptimizer optimizer;

    // Decision variables
    std::vector<HxExpression> bins;

    // Weight of each bin in the solution
    std::vector<HxExpression> binWeights;

    // Whether the bin is used in the solution
    std::vector<HxExpression> binsUsed;

    // Objective
    HxExpression totalBinsUsed;

public:
    /* Read instance data */
    void readInstance(const string& fileName) {
        ifstream infile;
        infile.exceptions(ifstream::failbit | ifstream::badbit);
        infile.open(fileName.c_str());

        infile >> nbItems;
        infile >> binCapacity;

        weightsData.resize(nbItems);
        for (int i = 0; i < nbItems; ++i) {
            infile >> weightsData[i];
        }

        nbMinBins = ceil(accumulate(weightsData.begin(), weightsData.end(), 0.0) / binCapacity);
        nbMaxBins = min(2 * nbMinBins, nbItems);
    }

    void solve(int limit) {
        // Declare the optimization model
        HxModel model = optimizer.getModel();

        bins.resize(nbMaxBins);
        binWeights.resize(nbMaxBins);
        binsUsed.resize(nbMaxBins);

        // Set decisions: bins[k] represents the items in bin k
        for (int k = 0; k < nbMaxBins; ++k) {
            bins[k] = model.setVar(nbItems);
        }

        // Each item must be in one bin and one bin only
        model.constraint(model.partition(bins.begin(), bins.end()));

        // Create an array and a function to retrieve the item's weight
        HxExpression weights = model.array(weightsData.begin(), weightsData.end());
        HxExpression weightLambda = model.createLambdaFunction([&](HxExpression i) { return weights[i]; });

        for (int k = 0; k < nbMaxBins; ++k) {
            // Weight constraint for each bin
            binWeights[k] = model.sum(bins[k], weightLambda);
            model.constraint(binWeights[k] <= binCapacity);

            // Bin k is used if at least one item is in it
            binsUsed[k] = model.count(bins[k]) > 0;
        }

        // Count the used bins
        totalBinsUsed = model.sum(binsUsed.begin(), binsUsed.end());

        // Minimize the number of used bins
        model.minimize(totalBinsUsed);

        model.close();

        // Parametrize the optimizer
        optimizer.getParam().setTimeLimit(limit);

        // Stop the search if the lower threshold is reached
        optimizer.getParam().setObjectiveThreshold(0, (hxint)nbMinBins);

        optimizer.solve();
    }

    /* Write the solution in a file */
    void writeSolution(const string& fileName) {
        ofstream outfile;
        outfile.exceptions(ofstream::failbit | ofstream::badbit);
        outfile.open(fileName.c_str());
        for (int k = 0; k < nbMaxBins; ++k) {
            if (binsUsed[k].getValue()) {
                outfile << "Bin weight: " << binWeights[k].getValue() << " | Items: ";
                HxCollection binCollection = bins[k].getCollectionValue();
                for (int i = 0; i < binCollection.count(); ++i) {
                    outfile << binCollection[i] << " ";
                }
                outfile << endl;
            }
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: bin_packing inputFile [outputFile] [timeLimit]" << endl;
        return 1;
    }

    const char* instanceFile = argv[1];
    const char* solFile = argc > 2 ? argv[2] : NULL;
    const char* strTimeLimit = argc > 3 ? argv[3] : "5";

    try {
        BinPacking model;
        model.readInstance(instanceFile);
        model.solve(atoi(strTimeLimit));
        if (solFile != NULL)
            model.writeSolution(solFile);
        return 0;
    } catch (const exception& e) {
        cerr << "An error occurred: " << e.what() << endl;
        return 1;
    }
}
