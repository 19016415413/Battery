%% ========================================================================
%  Battery State-of-Health (SOH) Prediction — Method Flow Pseudocode
%  Note: This file describes the overall pipeline structure only.
%  It does NOT contain executable implementation details
%  (feature-engineering formulas, exact hyperparameter grids, thresholds,
%  etc. have been abstracted out).
%  This file cannot be run directly — it is for illustrating the
%  research design and modeling workflow only.
%  The complete code integrates all the engineering files for dataset loading, model training, and result visualization. It also comes with an exclusive running environment configuration, a list of required packages, and step-by-step reproduction instructions. Due to the large number of files, they cannot be submitted as an attachment to the manuscript. If you need the complete source code project, please contact the corresponding author for it.
% ========================================================================

%% ---------------- [0] Initialization ----------------
PSEUDOCODE:
    Set random seed
    Define plotting color scheme (irrelevant to modeling logic, omitted)
    Define battery sample list = {Battery1, Battery2, Battery3, Battery4}


%% ---------------- [1] Data Loading & Feature Extraction ----------------
FUNCTION LoadBatteryData(batteryList):
    FOR EACH battery IN batteryList:
        rawCycleData = Load battery charge/discharge data file
        FOR EACH cycle IN rawCycleData:
            IF cycleType == "discharge":
                Extract voltage / current / temperature / time series for this cycle
                IF data has missing or non-finite values:
                    Skip this cycle
                ELSE:
                    <featureVector = ExtractFeatures(voltage, current, temperature, time)>
                    // ExtractFeatures internally includes:
                    // Aggregated statistics (mean/std/extrema, etc.)
                    // Power-derived quantities
                    // Impedance-related features (matched from the nearest
                    //   available impedance measurement record, if present)
                    Store featureVector, capacityLabel, cycleIndex
    RETURN {featureMatrix, capacityLabels, cycleIndices} organized per battery
END FUNCTION


%% ---------------- [2] Data Consolidation & Cleaning ----------------
PSEUDOCODE:
    Concatenate all batteries' feature matrices X_all, labels y_all, battery IDs battID

    Missing value handling:
        Compute missing-value ratio per feature
        Analyze missingness mechanism (concentrated in certain feature types?
            -> determine MCAR/MAR pattern)
        <Run a statistical test to check whether missingness relates to cycle stage>
        Drop rows containing missing values
        Report cleaning summary (sample counts before/after, retention rate)

    Integrity checks:
        Assert feature/label/batteryID dimensions match
        Assert cleaned data contains no non-finite values


%% ---------------- [3] Multicollinearity Diagnostics ----------------
FUNCTION ComputeVIF(X):
    Standardize X
    FOR EACH feature j:
        Regress feature j on all remaining features
        <Compute Variance Inflation Factor VIF_j from the regression fit quality>
    RETURN VIF vector
END FUNCTION

PSEUDOCODE:
    VIF = ComputeVIF(X_all)
    Flag features with VIF above threshold as "high collinearity"
    Report diagnostics (does not alter the modeling pipeline; for discussion only)


%% ---------------- [4] Unified Hyperparameter Search Space ----------------
PSEUDOCODE (exact values omitted, structure only):
    Define <Random Forest hyperparameter grid>     // e.g., candidate min-leaf sizes,
                                                    //   candidate feature-sampling ratios
    Define <LSBoost hyperparameter grid>           // e.g., tree depth, number of weak
                                                    //   learners, learning rate, min-leaf size
    Define <Lasso regularization coefficient candidates>   // log-scale sweep range
    Define <Ridge regularization coefficient candidates>   // log-scale sweep range
    Define <Random Forest number of trees (fixed value)>

    // Key design principle: the LOBO evaluation below and the final
    // model training must share the exact same search space, to keep
    // the evaluation protocol consistent.


%% ---------------- [5] Leave-One-Battery-Out Cross-Validation (LOBO) ----------------
FUNCTION LOBO_Evaluate(X_all, y_all, battID, modelList, searchSpace):
    FOR EACH testBattery IN batteryList:
        trainSet = all samples except testBattery
        testSet  = samples from testBattery

        innerCVFolds = split by remaining batteries (battery-level,
                       to prevent data leakage)

        FOR EACH model IN modelList:
            bestParams = NULL
            bestScore = +INF

            FOR EACH candidateHyperparamSet IN searchSpace[model]:
                FOR EACH innerFold:
                    <Standardize independently within this fold
                     (mean/std computed only from the fold's training portion)>
                    <Train a temporary model with the candidate hyperparameters>
                    <Predict on the fold's validation set and compute RMSE>
                Average RMSE across inner folds = score for this hyperparameter set
                IF score < bestScore:
                    Update bestParams, bestScore

            <Retrain the model on the full training set for this LOBO
             configuration using bestParams>
            <Predict on testBattery's test set>
            Record RMSE, R2, MAE, etc.

    RETURN performance metrics for each model across the 4 LOBO configurations
END FUNCTION

PSEUDOCODE:
    loboResults = LOBO_Evaluate(X_all, y_all, battID,
                                 {OptimizedRandomForest, LSBoost, Lasso, Ridge},
                                 unifiedHyperparamGrid)


%% ---------------- [6] Main Train/Test Split & Leakage Verification ----------------
PSEUDOCODE:
    Select one battery as the final held-out test set
    Remaining batteries form the training set

    Four data-leakage checks:
        Check 1: no overlap between train/test sample indices
        Check 2: no overlap between train/test battery IDs
        Check 3: standardization parameters derived only from the training set
        Check 4: each inner CV fold re-standardizes independently

    IF any check fails:
        Raise an error and halt the pipeline


%% ---------------- [7] Hyperparameter Tuning (Main Training Set) ----------------
PSEUDOCODE, repeated for each model with the same structure:

FUNCTION TuneModel(modelType, X_train_raw, y_train, cvFolds, searchSpace):
    FOR EACH candidateHyperparamSet IN searchSpace[modelType]:
        FOR EACH fold IN cvFolds:
            <Standardize independently within the fold>
            <Train a temporary model>
            <Predict on validation set -> compute RMSE>
        Record the average CV-RMSE for this hyperparameter set
    Select the hyperparameter set with the lowest CV-RMSE
    RETURN bestParams, bestCV_RMSE
END FUNCTION

Calls:
    bestParams_RF      = TuneModel(<RandomForest>, ...)
    bestParams_LSBoost = TuneModel(<LSBoost>, ...)
    bestParams_Lasso   = TuneModel(<Lasso>, ...)
    bestParams_Ridge   = TuneModel(<Ridge>, ...)


%% ---------------- [8] Final Model Training & Prediction ----------------
PSEUDOCODE:
    Standardize train/test sets using training-set-derived parameters

    FOR EACH model IN {OptimizedRandomForest, LSBoost, Lasso, Ridge}:
        <Train final model on the full training set using its best hyperparameters>
        <Predict on both training set and test set>
        Compute train/test RMSE, R2, MAE, MAPE


%% ---------------- [9] Statistical Significance Testing ----------------
PSEUDOCODE:
    Based on LOBO results (4 battery-level observations per model):
        <Run a Friedman test to assess whether overall performance
         differences across models are significant>
        IF significant:
            <Run a Nemenyi post-hoc test for pairwise model comparisons>

    For each model's residuals:
        <Run a normality test (e.g., Shapiro-Wilk or equivalent)>
        Plot Q-Q plots to aid visual assessment


%% ---------------- [10] Feature Importance Analysis ----------------
FUNCTION PermutationImportance(model, X_test, y_test):
    baselineError = compute model error on the original test set
    FOR EACH feature j:
        X_permuted = copy of X_test with column j randomly shuffled
        permutedError = compute model error on X_permuted
        importance(j) = permutedError - baselineError
    RETURN importance vector
END FUNCTION

PSEUDOCODE:
    FOR EACH model IN {OptimizedRandomForest, LSBoost, Lasso, Ridge}:
        importance[model] = PermutationImportance(model, X_test, y_test)

    consensusImportance = average of all models' importance vectors
    // Note: this is permutation importance, NOT SHAP values


%% ---------------- [11] Results Aggregation & Visualization ----------------
PSEUDOCODE:
    Aggregate metrics: RMSE / R2 / MAE / MAPE (train and test)
    Compute generalization gap = |train RMSE - test RMSE|
    Compute overall score = weighted average of normalized metrics

    Generate 16 figures, including but not limited to:
        - Predicted vs. actual scatter plots
        - Inter-model prediction disagreement band
        - Feature correlation ranking
        - Generalization gap comparison
        - Combined metrics comparison and efficiency-vs-performance plot
        - Feature importance ranking chart
    (Specific plotting code is presentation-only and unrelated to the
     core modeling logic; omitted here.)


%% ---------------- [12] Output & Conclusions ----------------
PSEUDOCODE:
    Select the best model based on test-set RMSE
    Report LOBO results summary table (mean +/- std)
    Report methodological notes and limitations
    (small sample size, single battery chemistry, etc.)

%% ========================================================================
%  END OF PSEUDOCODE
%  Note: This file describes the pipeline structure only. It does not
%  contain the specific parameters or implementation details needed
%  to reproduce actual results.
% ========================================================================
