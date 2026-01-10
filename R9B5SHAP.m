clear; clc; close all;

rng(42, 'twister');

natureColors = struct();
natureColors.deepRed      = [215/255, 48/255, 39/255];
natureColors.brightRed    = [252/255, 141/255, 89/255];
natureColors.lightOrange  = [254/255, 224/255, 139/255];
natureColors.paleYellow   = [255/255, 255/255, 191/255];
natureColors.lightBlue    = [173/255, 216/255, 230/255];  
natureColors.skyBlue      = [145/255, 191/255, 219/255];
natureColors.deepBlue     = [69/255, 117/255, 180/255];
natureColors.orange       = [253/255, 174/255, 97/255];
natureColors.yellow       = [254/255, 237/255, 160/255];
natureColors.cyan         = [171/255, 217/255, 233/255];
natureColors.charcoal     = [50/255, 50/255, 50/255];
natureColors.mediumGray   = [120/255, 120/255, 120/255];
natureColors.lightGray    = [180/255, 180/255, 180/255];
natureColors.purple       = [158/255, 154/255, 200/255];
natureColors.green        = [102/255, 194/255, 165/255];
natureColors.teal         = [0/255, 128/255, 128/255];
natureColors.lightGreen   = [217/255, 239/255, 139/255];  

batteryColors = [
    natureColors.deepRed;
    natureColors.orange;
    natureColors.yellow;
    natureColors.skyBlue
];

modelColors = struct();
modelColors.optrf  = natureColors.orange;
modelColors.lsb    = natureColors.deepBlue;
modelColors.lasso  = natureColors.cyan;
modelColors.ridge  = natureColors.purple;

set(0, 'DefaultAxesFontName', 'Helvetica');
set(0, 'DefaultAxesFontSize', 9);
set(0, 'DefaultTextFontName', 'Helvetica');
set(0, 'DefaultLineLineWidth', 1.5);
set(0, 'DefaultAxesLineWidth', 1);
set(0, 'DefaultAxesBox', 'on');

batteries = {'B0005', 'B0006', 'B0007', 'B0018'};
files     = {'B0005.mat', 'B0006.mat', 'B0007.mat', 'B0018.mat'};

fprintf('========================================\n');
fprintf('CORRECTED VERSION V5 - PUBLICATION READY\n');
fprintf('Models: Opt RF | LSBoost | Lasso | Ridge\n');
fprintf('========================================\n\n');
fprintf('Loading battery data...\n');
allData = struct();

for b = 1:length(batteries)
    if exist(files{b}, 'file')
        S = load(files{b});
        if ~isfield(S, batteries{b})
            warning('  %s loaded but variable %s not found', files{b}, batteries{b});
            continue;
        end
        battData = S.(batteries{b});
        nCycles  = numel(battData.cycle);
        features = [];
        capacity = [];
        cycleNum = [];
        dischargeIdx = 1;
        
        for i = 1:nCycles
            if strcmpi(battData.cycle(i).type, 'discharge')
                if ~isfield(battData.cycle(i), 'data'), continue; end
                D = battData.cycle(i).data;
                needed = {'Capacity','Voltage_measured','Current_measured','Temperature_measured','Time'};
                if ~all(isfield(D, needed)), continue; end

                cap  = double(D.Capacity);
                V    = double(D.Voltage_measured(:));
                I    = double(D.Current_measured(:));
                T    = double(D.Temperature_measured(:));
                Time = double(D.Time(:));
                
                if any([isempty(V), isempty(I), isempty(T), isempty(Time), isempty(cap)])
                    continue;
                end
                
                if any(~isfinite(V)) || any(~isfinite(I)) || any(~isfinite(T))
                    continue;
                end

                feat = [dischargeIdx, ...
                    mean(V), std(V), min(V), max(V), ...
                    mean(abs(I)), std(I), ...
                    mean(T), max(T), (max(T)-min(T)), ...
                    mean(V .* abs(I)), max(V .* abs(I))];

                impIdx = find(strcmp({battData.cycle(1:i).type}, 'impedance'), 1, 'last');
                if ~isempty(impIdx) && isfield(battData.cycle(impIdx), 'data')
                    Idata = battData.cycle(impIdx).data;
                    Re  = NaN; Rct = NaN;
                    if isfield(Idata, 'Re')  && ~isempty(Idata.Re),  Re  = double(Idata.Re(1));  end
                    if isfield(Idata, 'Rct') && ~isempty(Idata.Rct), Rct = double(Idata.Rct(1)); end
                else
                    Re = NaN; Rct = NaN;
                end

                feat = [feat, Re, Rct];
                features = [features; feat];
                capacity = [capacity; cap];
                cycleNum = [cycleNum; dischargeIdx];
                dischargeIdx = dischargeIdx + 1;
            end
        end

        allData.(batteries{b}).features = features;
        allData.(batteries{b}).capacity = capacity;
        allData.(batteries{b}).cycleNum = cycleNum;
        fprintf('  %s: %d discharge cycles\n', batteries{b}, numel(capacity));
    else
        error('File %s not found. Please ensure data files are in the current directory.', files{b});
    end
end

fprintf('\nPreparing training data...\n');
X_all  = [];
y_all  = [];
battID = [];
cycleID_all = [];

for b = 1:length(batteries)
    if isfield(allData, batteries{b}) && ~isempty(allData.(batteries{b}).capacity)
        X_all  = [X_all; allData.(batteries{b}).features];
        y_all  = [y_all; allData.(batteries{b}).capacity];
        battID = [battID; b * ones(numel(allData.(batteries{b}).capacity), 1)];
        cycleID_all = [cycleID_all; allData.(batteries{b}).cycleNum];
    end
end

n_total_original = size(X_all, 1);
nanRows = any(isnan(X_all), 2) | isnan(y_all);
n_removed = sum(nanRows);
pct_removed = 100 * n_removed / n_total_original;

fprintf('\n=== Data Cleaning Report ===\n');
fprintf('Total samples (before cleaning): %d\n', n_total_original);
fprintf('Samples with missing values: %d (%.2f%%)\n', n_removed, pct_removed);

featureNames = {'Cycle','V_{mean}','V_{std}','V_{min}','V_{max}', ...
    'I_{mean}','I_{std}','T_{mean}','T_{max}','\DeltaT', ...
    'P_{mean}','P_{max}','R_e','R_{ct}'};

fprintf('Missing data pattern analysis:\n');
missing_by_feature = zeros(length(featureNames), 1);
for i = 1:length(featureNames)
    n_missing = sum(isnan(X_all(:,i)));
    missing_by_feature(i) = n_missing;
    if n_missing > 0
        fprintf('  %s: %d missing (%.1f%%)\n', featureNames{i}, n_missing, ...
            100*n_missing/n_total_original);
    end
end

fprintf('\nMissing Data Mechanism Analysis:\n');
if all(missing_by_feature(1:12) == 0) && any(missing_by_feature(13:14) > 0)
    fprintf('   - Missingness confined to impedance features (R_e, R_ct)\n');
    fprintf('   - Non-impedance features have no missing values\n');
    
    if sum(nanRows) > 0
        missing_cycles = X_all(nanRows, 1);
        missing_capacity = y_all(nanRows);
        complete_cycles = X_all(~nanRows, 1);
        
        valid_missing_cycles = missing_cycles(~isnan(missing_cycles));
        if ~isempty(valid_missing_cycles) && length(complete_cycles) > 1
            fprintf('   - Missing data cycle range: [%.0f, %.0f]\n', ...
                min(valid_missing_cycles), max(valid_missing_cycles));
            
            try
                [~, p_ttest] = ttest2(complete_cycles, valid_missing_cycles);
                fprintf('   - Cycle difference test (t-test p-value): %.4f\n', p_ttest);
                
                if p_ttest > 0.05
                    fprintf('   - No significant cycle difference -> suggests MCAR\n');
                else
                    fprintf('   - Significant cycle difference -> suggests MAR\n');
                end
            catch
                fprintf('   - Statistical test could not be performed (insufficient data)\n');
            end
        end
    end
    
    fprintf('   - Pattern analysis: Impedance measurements are optional periodic tests\n');
    fprintf('   - Conclusion: Missing mechanism likely MAR/MCAR\n');
else
    fprintf('   - Missingness pattern is complex across multiple features\n');
    fprintf('   - Further investigation required\n');
end

X_all(nanRows, :) = [];
y_all(nanRows)    = [];
battID(nanRows)   = [];
cycleID_all(nanRows) = [];

fprintf('Final dataset: %d samples (%.1f%% retained)\n', size(X_all, 1), 100*(1-pct_removed/100));

assert(size(X_all, 1) == length(y_all), 'Features and labels size mismatch!');
assert(size(X_all, 1) == length(battID), 'Features and battery ID size mismatch!');
assert(all(isfinite(X_all(:))), 'X_all contains non-finite values after cleaning!');
assert(all(isfinite(y_all)), 'y_all contains non-finite values after cleaning!');

fprintf('\n=== Multicollinearity Check (VIF) ===\n');
n_features = size(X_all, 2);
VIF = zeros(n_features, 1);
X_temp = X_all;
X_temp_mean = mean(X_temp, 1);
X_temp_std = std(X_temp, 0, 1);
X_temp_std(X_temp_std < 1e-10) = 1;
X_temp_norm = (X_temp - X_temp_mean) ./ X_temp_std;

for j = 1:n_features
    y_vif = X_temp_norm(:, j);
    X_vif = X_temp_norm(:, setdiff(1:n_features, j));
    X_vif_with_intercept = [ones(size(X_vif, 1), 1), X_vif];
    b = X_vif_with_intercept \ y_vif;
    y_pred_vif = X_vif_with_intercept * b;
    
    SS_res = sum((y_vif - y_pred_vif).^2);
    SS_tot = sum((y_vif - mean(y_vif)).^2);
    
    if SS_tot > 1e-10
        R2_vif = 1 - SS_res / SS_tot;
        R2_vif = max(0, min(R2_vif, 1 - 1e-10));
        VIF(j) = 1 / (1 - R2_vif);
    else
        VIF(j) = 1;
    end
end

fprintf('Variance Inflation Factors:\n');
high_vif_count = 0;
for j = 1:n_features
    vif_status = '';
    if VIF(j) > 10
        vif_status = ' HIGH';
        high_vif_count = high_vif_count + 1;
    elseif VIF(j) > 5
        vif_status = ' (moderate)';
    end
    fprintf('  %s: %.2f%s\n', featureNames{j}, VIF(j), vif_status);
end

if high_vif_count > 0
    fprintf('\nWARNING: %d features have VIF > 10 (severe multicollinearity)\n', high_vif_count);
else
    fprintf('No severe multicollinearity detected (all VIF < 10)\n');
end

fprintf('\n=== Unified Hyperparameter Search Space ===\n');
rf_minLeafSizes_unified = [5, 8, 10, 15, 20];
rf_mtry_ratios_unified = [1.0, 0.75, 0.5, 0.33];
lsb_depths_unified = [1, 2, 3];
lsb_learners_unified = [100, 150, 200];
lsb_learnRates_unified = [0.1, 0.15, 0.2];
lsb_minLeafSizes_unified = [5, 8, 10];
lasso_lambdas_unified = logspace(-6, -1, 30);
ridge_lambdas_unified = logspace(-3, 3, 30);
rf_nTrees = 300;

fprintf('RF: MinLeafSizes=%s, mtry_ratios=%s\n', mat2str(rf_minLeafSizes_unified), mat2str(rf_mtry_ratios_unified));
fprintf('LSBoost: Depths=%s, Learners=%s, LRs=%s\n', mat2str(lsb_depths_unified), mat2str(lsb_learners_unified), mat2str(lsb_learnRates_unified));
fprintf('Lasso: %d lambda values from %.1e to %.1e\n', length(lasso_lambdas_unified), min(lasso_lambdas_unified), max(lasso_lambdas_unified));
fprintf('Ridge: %d lambda values from %.1e to %.1e\n', length(ridge_lambdas_unified), min(ridge_lambdas_unified), max(ridge_lambdas_unified));

fprintf('\n=== LOBO for ALL Models (Unified Parameters) ===\n');
fprintf('Method: Each CV fold re-normalizes independently\n\n');

models_list = {'OptRF', 'LSBoost', 'Lasso', 'Ridge'};
model_names_display = {'Opt RF', 'LSBoost', 'Lasso', 'Ridge'};

lobo_results_all = struct();
for m = 1:4
    lobo_results_all.(models_list{m}).rmse = zeros(4, 1);
    lobo_results_all.(models_list{m}).r2 = zeros(4, 1);
    lobo_results_all.(models_list{m}).mae = zeros(4, 1);
end

for test_batt_id = 1:4
    fprintf('>> LOBO Configuration %d/4: Test Battery = %s\n', test_batt_id, batteries{test_batt_id});
    
    trainIdx_lobo = battID ~= test_batt_id;
    testIdx_lobo  = battID == test_batt_id;
    
    X_train_lobo_raw = X_all(trainIdx_lobo, :);
    X_test_lobo_raw  = X_all(testIdx_lobo, :);
    y_train_lobo = y_all(trainIdx_lobo);
    y_test_lobo  = y_all(testIdx_lobo);
    
    trainBattID_lobo = battID(trainIdx_lobo);
    cvBatteries_lobo = setdiff([1,2,3,4], test_batt_id);
    nFolds_lobo = length(cvBatteries_lobo);
    
    X_mean_lobo = mean(X_train_lobo_raw, 1);
    X_std_lobo  = std(X_train_lobo_raw, 0, 1);
    X_std_lobo(X_std_lobo < 1e-10) = 1;
    
    X_train_lobo = (X_train_lobo_raw - X_mean_lobo) ./ X_std_lobo;
    X_test_lobo  = (X_test_lobo_raw - X_mean_lobo) ./ X_std_lobo;
    
    best_rf_cv = inf;
    best_rf_params = struct('minLeaf', 5, 'mtry', 7);
    
    for ml = 1:length(rf_minLeafSizes_unified)
        for mt = 1:length(rf_mtry_ratios_unified)
            cv_scores = zeros(nFolds_lobo, 1);
            mtry = max(1, round(size(X_train_lobo_raw, 2) * rf_mtry_ratios_unified(mt)));
            
            for fold = 1:nFolds_lobo
                fold_train_idx = trainBattID_lobo ~= cvBatteries_lobo(fold);
                fold_val_idx = trainBattID_lobo == cvBatteries_lobo(fold);
                
                X_fold_train_raw = X_train_lobo_raw(fold_train_idx, :);
                X_fold_val_raw = X_train_lobo_raw(fold_val_idx, :);
                
                X_fold_mean = mean(X_fold_train_raw, 1);
                X_fold_std = std(X_fold_train_raw, 0, 1);
                X_fold_std(X_fold_std < 1e-10) = 1;
                
                X_fold_train = (X_fold_train_raw - X_fold_mean) ./ X_fold_std;
                X_fold_val = (X_fold_val_raw - X_fold_mean) ./ X_fold_std;
                
                rf_temp = TreeBagger(100, X_fold_train, y_train_lobo(fold_train_idx), ...
                    'Method', 'regression', 'MinLeafSize', rf_minLeafSizes_unified(ml), ...
                    'NumPredictorsToSample', mtry);
                y_pred_val = predict(rf_temp, X_fold_val);
                cv_scores(fold) = sqrt(mean((y_train_lobo(fold_val_idx) - y_pred_val).^2));
            end
            
            if mean(cv_scores) < best_rf_cv
                best_rf_cv = mean(cv_scores);
                best_rf_params.minLeaf = rf_minLeafSizes_unified(ml);
                best_rf_params.mtry = mtry;
            end
        end
    end
    
    best_lsb_cv = inf;
    best_lsb_params = struct('depth', 1, 'learners', 100, 'learnRate', 0.1, 'minLeaf', 5);
    
    for d = 1:length(lsb_depths_unified)
        for n = 1:length(lsb_learners_unified)
            for lr = 1:length(lsb_learnRates_unified)
                for ml = 1:length(lsb_minLeafSizes_unified)
                    cv_scores = zeros(nFolds_lobo, 1);
                    
                    for fold = 1:nFolds_lobo
                        fold_train_idx = trainBattID_lobo ~= cvBatteries_lobo(fold);
                        fold_val_idx = trainBattID_lobo == cvBatteries_lobo(fold);
                        
                        X_fold_train_raw = X_train_lobo_raw(fold_train_idx, :);
                        X_fold_val_raw = X_train_lobo_raw(fold_val_idx, :);
                        
                        X_fold_mean = mean(X_fold_train_raw, 1);
                        X_fold_std = std(X_fold_train_raw, 0, 1);
                        X_fold_std(X_fold_std < 1e-10) = 1;
                        
                        X_fold_train = (X_fold_train_raw - X_fold_mean) ./ X_fold_std;
                        X_fold_val = (X_fold_val_raw - X_fold_mean) ./ X_fold_std;
                        
                        weakLearner = templateTree('MaxNumSplits', ...
                            2^lsb_depths_unified(d) - 1, ...
                            'MinLeafSize', lsb_minLeafSizes_unified(ml));
                        mdl_temp = fitrensemble(X_fold_train, ...
                            y_train_lobo(fold_train_idx), 'Method', 'LSBoost', ...
                            'NumLearningCycles', lsb_learners_unified(n), ...
                            'Learners', weakLearner, ...
                            'LearnRate', lsb_learnRates_unified(lr));
                        
                        y_pred_val = predict(mdl_temp, X_fold_val);
                        cv_scores(fold) = sqrt(mean((y_train_lobo(fold_val_idx) - y_pred_val).^2));
                    end
                    
                    if mean(cv_scores) < best_lsb_cv
                        best_lsb_cv = mean(cv_scores);
                        best_lsb_params.depth = lsb_depths_unified(d);
                        best_lsb_params.learners = lsb_learners_unified(n);
                        best_lsb_params.learnRate = lsb_learnRates_unified(lr);
                        best_lsb_params.minLeaf = lsb_minLeafSizes_unified(ml);
                    end
                end
            end
        end
    end
    
    best_lasso_cv = inf;
    best_lasso_lambda = 0.001;
    
    for lambda_idx = 1:length(lasso_lambdas_unified)
        cv_scores = zeros(nFolds_lobo, 1);
        
        for fold = 1:nFolds_lobo
            fold_train_idx = trainBattID_lobo ~= cvBatteries_lobo(fold);
            fold_val_idx = trainBattID_lobo == cvBatteries_lobo(fold);
            
            X_fold_train_raw = X_train_lobo_raw(fold_train_idx, :);
            X_fold_val_raw = X_train_lobo_raw(fold_val_idx, :);
            
            X_fold_mean = mean(X_fold_train_raw, 1);
            X_fold_std = std(X_fold_train_raw, 0, 1);
            X_fold_std(X_fold_std < 1e-10) = 1;
            
            X_fold_train = (X_fold_train_raw - X_fold_mean) ./ X_fold_std;
            X_fold_val = (X_fold_val_raw - X_fold_mean) ./ X_fold_std;
            
            [B_fold, FitInfo_fold] = lasso(X_fold_train, ...
                y_train_lobo(fold_train_idx), 'Lambda', lasso_lambdas_unified(lambda_idx));
            y_pred_val = X_fold_val * B_fold + FitInfo_fold.Intercept;
            cv_scores(fold) = sqrt(mean((y_train_lobo(fold_val_idx) - y_pred_val).^2));
        end
        
        if mean(cv_scores) < best_lasso_cv
            best_lasso_cv = mean(cv_scores);
            best_lasso_lambda = lasso_lambdas_unified(lambda_idx);
        end
    end
    
    best_ridge_cv = inf;
    best_ridge_lambda = 1;
    
    for lambda_idx = 1:length(ridge_lambdas_unified)
        cv_scores = zeros(nFolds_lobo, 1);
        
        for fold = 1:nFolds_lobo
            fold_train_idx = trainBattID_lobo ~= cvBatteries_lobo(fold);
            fold_val_idx = trainBattID_lobo == cvBatteries_lobo(fold);
            
            X_fold_train_raw = X_train_lobo_raw(fold_train_idx, :);
            X_fold_val_raw = X_train_lobo_raw(fold_val_idx, :);
            
            X_fold_mean = mean(X_fold_train_raw, 1);
            X_fold_std = std(X_fold_train_raw, 0, 1);
            X_fold_std(X_fold_std < 1e-10) = 1;
            
            X_fold_train = (X_fold_train_raw - X_fold_mean) ./ X_fold_std;
            X_fold_val = (X_fold_val_raw - X_fold_mean) ./ X_fold_std;
            
            B = ridge(y_train_lobo(fold_train_idx), X_fold_train, ridge_lambdas_unified(lambda_idx), 0);
            y_pred_val = [ones(sum(fold_val_idx),1), X_fold_val] * B;
            cv_scores(fold) = sqrt(mean((y_train_lobo(fold_val_idx) - y_pred_val).^2));
        end
        
        if mean(cv_scores) < best_ridge_cv
            best_ridge_cv = mean(cv_scores);
            best_ridge_lambda = ridge_lambdas_unified(lambda_idx);
        end
    end
    
    rf_lobo = TreeBagger(rf_nTrees, X_train_lobo, y_train_lobo, 'Method', 'regression', ...
        'MinLeafSize', best_rf_params.minLeaf, 'NumPredictorsToSample', best_rf_params.mtry);
    y_pred_rf = predict(rf_lobo, X_test_lobo);
    
    weakLearner = templateTree('MaxNumSplits', 2^best_lsb_params.depth - 1, ...
        'MinLeafSize', best_lsb_params.minLeaf);
    lsb_lobo = fitrensemble(X_train_lobo, y_train_lobo, 'Method', 'LSBoost', ...
        'NumLearningCycles', best_lsb_params.learners, 'Learners', weakLearner, ...
        'LearnRate', best_lsb_params.learnRate);
    y_pred_lsb = predict(lsb_lobo, X_test_lobo);
    
    [B_lasso, FitInfo_lasso] = lasso(X_train_lobo, y_train_lobo, 'Lambda', best_lasso_lambda);
    y_pred_lasso = X_test_lobo * B_lasso + FitInfo_lasso.Intercept;
    
    B_ridge = ridge(y_train_lobo, X_train_lobo, best_ridge_lambda, 0);
    y_pred_ridge = [ones(size(X_test_lobo,1),1), X_test_lobo] * B_ridge;
    
    y_preds_lobo = {y_pred_rf, y_pred_lsb, y_pred_lasso, y_pred_ridge};
    
    for m = 1:4
        rmse_val = sqrt(mean((y_test_lobo - y_preds_lobo{m}).^2));
        ss_res = sum((y_test_lobo - y_preds_lobo{m}).^2);
        ss_tot = sum((y_test_lobo - mean(y_test_lobo)).^2);
        r2_val = 1 - ss_res / ss_tot;
        mae_val = mean(abs(y_test_lobo - y_preds_lobo{m}));
        
        lobo_results_all.(models_list{m}).rmse(test_batt_id) = rmse_val;
        lobo_results_all.(models_list{m}).r2(test_batt_id) = r2_val;
        lobo_results_all.(models_list{m}).mae(test_batt_id) = mae_val;
    end
    
    fprintf('   Test Results:\n');
    for m = 1:4
        fprintf('     %s: RMSE=%.4f, R2=%.4f, MAE=%.4f\n', model_names_display{m}, ...
            lobo_results_all.(models_list{m}).rmse(test_batt_id), ...
            lobo_results_all.(models_list{m}).r2(test_batt_id), ...
            lobo_results_all.(models_list{m}).mae(test_batt_id));
    end
    fprintf('\n');
end

fprintf('=== LOBO Summary (All Models) ===\n');
fprintf('%-10s | %-20s | %-20s | %-20s\n', 'Model', 'RMSE (mean+/-std)', 'R2 (mean+/-std)', 'MAE (mean+/-std)');
fprintf('-----------|----------------------|----------------------|---------------------\n');
for m = 1:4
    rmse_vals = lobo_results_all.(models_list{m}).rmse;
    r2_vals = lobo_results_all.(models_list{m}).r2;
    mae_vals = lobo_results_all.(models_list{m}).mae;
    fprintf('%-10s | %.4f +/- %.4f      | %.4f +/- %.4f      | %.4f +/- %.4f\n', ...
        model_names_display{m}, mean(rmse_vals), std(rmse_vals), ...
        mean(r2_vals), std(r2_vals), mean(mae_vals), std(mae_vals));
end
fprintf('\n');

testBatteryID = 4;
trainIdx = battID ~= testBatteryID;
testIdx  = battID == testBatteryID;

X_train_raw = X_all(trainIdx, :);
X_test_raw  = X_all(testIdx, :);
y_train = y_all(trainIdx);
y_test  = y_all(testIdx);
trainBattID = battID(trainIdx);
trainCycleID = cycleID_all(trainIdx);

X_mean = mean(X_train_raw, 1);
X_std  = std(X_train_raw, 0, 1);
X_std(X_std < 1e-10) = 1;

X_train = (X_train_raw - X_mean) ./ X_std;
X_test  = (X_test_raw - X_mean) ./ X_std;

fprintf('Training: %d samples | Testing: %d samples\n', sum(trainIdx), sum(testIdx));
fprintf('Test battery: %s\n', batteries{testBatteryID});

fprintf('\n=== Data Leakage Verification ===\n');
train_samples = find(trainIdx);
test_samples = find(testIdx);
overlap = intersect(train_samples, test_samples);

if isempty(overlap)
    fprintf('Check 1 PASSED: No overlap between train and test sets\n');
else
    error('CRITICAL ERROR: %d samples overlap between train and test!', length(overlap));
end

train_batteries = unique(battID(trainIdx));
test_batteries = unique(battID(testIdx));
battery_overlap = intersect(train_batteries, test_batteries);

if isempty(battery_overlap)
    fprintf('Check 2 PASSED: Battery-level separation maintained\n');
else
    error('CRITICAL ERROR: Battery overlap detected!');
end

fprintf('Check 3 PASSED: Normalization from training set only\n');
fprintf('Check 4: CV will re-normalize each fold independently\n\n');

cvBatteries = setdiff([1,2,3,4], testBatteryID);
nFolds = length(cvBatteries);
cvFolds = cell(nFolds, 1);
for fold = 1:nFolds
    cvFolds{fold}.train = trainBattID ~= cvBatteries(fold);
    cvFolds{fold}.val = trainBattID == cvBatteries(fold);
end

fprintf('Cross-validation: %d folds (battery-level)\n\n', nFolds);

fprintf('=== Hyperparameter Tuning (Unified Search Space) ===\n');

fprintf('\n[1/4] Optimizing Random Forest...\n');
rf_cv_results = zeros(length(rf_minLeafSizes_unified), length(rf_mtry_ratios_unified));

for i = 1:length(rf_minLeafSizes_unified)
    for j = 1:length(rf_mtry_ratios_unified)
        minLeaf = rf_minLeafSizes_unified(i);
        mtry = max(1, round(size(X_train_raw, 2) * rf_mtry_ratios_unified(j)));
        cv_scores = zeros(nFolds, 1);
        
        for fold = 1:nFolds
            X_fold_train_raw = X_train_raw(cvFolds{fold}.train, :);
            X_fold_val_raw = X_train_raw(cvFolds{fold}.val, :);
            
            X_fold_mean = mean(X_fold_train_raw, 1);
            X_fold_std = std(X_fold_train_raw, 0, 1);
            X_fold_std(X_fold_std < 1e-10) = 1;
            
            X_fold_train = (X_fold_train_raw - X_fold_mean) ./ X_fold_std;
            X_fold_val = (X_fold_val_raw - X_fold_mean) ./ X_fold_std;
            
            rf_temp = TreeBagger(rf_nTrees, X_fold_train, ...
                y_train(cvFolds{fold}.train), 'Method', 'regression', ...
                'MinLeafSize', minLeaf, 'NumPredictorsToSample', mtry);
            y_pred = predict(rf_temp, X_fold_val);
            cv_scores(fold) = sqrt(mean((y_train(cvFolds{fold}.val) - y_pred).^2));
        end
        rf_cv_results(i, j) = mean(cv_scores);
    end
end
[min_rf_rmse, min_idx] = min(rf_cv_results(:));
[best_i, best_j] = ind2sub(size(rf_cv_results), min_idx);
best_rf_minLeaf = rf_minLeafSizes_unified(best_i);
best_rf_mtry = max(1, round(size(X_train_raw, 2) * rf_mtry_ratios_unified(best_j)));
fprintf('Opt RF: MinLeaf=%d, mtry=%d, CV-RMSE=%.4f\n', best_rf_minLeaf, best_rf_mtry, min_rf_rmse);

fprintf('\n[2/4] Optimizing LSBoost...\n');
lsb_cv_results = zeros(length(lsb_depths_unified), length(lsb_learners_unified), ...
    length(lsb_learnRates_unified), length(lsb_minLeafSizes_unified));

for i = 1:length(lsb_depths_unified)
    for j = 1:length(lsb_learners_unified)
        for k = 1:length(lsb_learnRates_unified)
            for m = 1:length(lsb_minLeafSizes_unified)
                cv_scores = zeros(nFolds, 1);
                
                for fold = 1:nFolds
                    X_fold_train_raw = X_train_raw(cvFolds{fold}.train, :);
                    X_fold_val_raw = X_train_raw(cvFolds{fold}.val, :);
                    
                    X_fold_mean = mean(X_fold_train_raw, 1);
                    X_fold_std = std(X_fold_train_raw, 0, 1);
                    X_fold_std(X_fold_std < 1e-10) = 1;
                    
                    X_fold_train = (X_fold_train_raw - X_fold_mean) ./ X_fold_std;
                    X_fold_val = (X_fold_val_raw - X_fold_mean) ./ X_fold_std;
                    
                    weakLearner = templateTree('MaxNumSplits', 2^lsb_depths_unified(i) - 1, ...
                        'MinLeafSize', lsb_minLeafSizes_unified(m));
                    mdl = fitrensemble(X_fold_train, ...
                        y_train(cvFolds{fold}.train), 'Method', 'LSBoost', ...
                        'NumLearningCycles', lsb_learners_unified(j), 'Learners', weakLearner, ...
                        'LearnRate', lsb_learnRates_unified(k));
                    y_pred = predict(mdl, X_fold_val);
                    cv_scores(fold) = sqrt(mean((y_train(cvFolds{fold}.val) - y_pred).^2));
                end
                lsb_cv_results(i, j, k, m) = mean(cv_scores);
            end
        end
    end
end
[min_lsb_rmse, min_idx] = min(lsb_cv_results(:));
[best_i, best_j, best_k, best_m] = ind2sub(size(lsb_cv_results), min_idx);
best_lsb_depth = lsb_depths_unified(best_i);
best_lsb_learners = lsb_learners_unified(best_j);
best_lsb_learnRate = lsb_learnRates_unified(best_k);
best_lsb_minLeaf = lsb_minLeafSizes_unified(best_m);
fprintf('LSBoost: Depth=%d, Learners=%d, LR=%.2f, MinLeaf=%d, CV-RMSE=%.4f\n', ...
    best_lsb_depth, best_lsb_learners, best_lsb_learnRate, best_lsb_minLeaf, min_lsb_rmse);

fprintf('\n[3/4] Optimizing Lasso...\n');
lasso_cv_scores_per_lambda = zeros(length(lasso_lambdas_unified), 1);

for lambda_idx = 1:length(lasso_lambdas_unified)
    current_lambda = lasso_lambdas_unified(lambda_idx);
    cv_scores = zeros(nFolds, 1);
    
    for fold = 1:nFolds
        X_fold_train_raw = X_train_raw(cvFolds{fold}.train, :);
        X_fold_val_raw = X_train_raw(cvFolds{fold}.val, :);
        
        X_fold_mean = mean(X_fold_train_raw, 1);
        X_fold_std = std(X_fold_train_raw, 0, 1);
        X_fold_std(X_fold_std < 1e-10) = 1;
        
        X_fold_train = (X_fold_train_raw - X_fold_mean) ./ X_fold_std;
        X_fold_val = (X_fold_val_raw - X_fold_mean) ./ X_fold_std;
        
        [B_fold, FitInfo_fold] = lasso(X_fold_train, ...
            y_train(cvFolds{fold}.train), 'Lambda', current_lambda);
        y_pred_val = X_fold_val * B_fold + FitInfo_fold.Intercept;
        cv_scores(fold) = sqrt(mean((y_train(cvFolds{fold}.val) - y_pred_val).^2));
    end
    
    lasso_cv_scores_per_lambda(lambda_idx) = mean(cv_scores);
end

[lasso_cv_rmse, best_lambda_idx] = min(lasso_cv_scores_per_lambda);
best_lambda_lasso_final = lasso_lambdas_unified(best_lambda_idx);
fprintf('Lasso: lambda=%.6f, CV-RMSE=%.4f\n', best_lambda_lasso_final, lasso_cv_rmse);

fprintf('\n[4/4] Optimizing Ridge...\n');
ridge_cv_results = zeros(length(ridge_lambdas_unified), 1);

for i = 1:length(ridge_lambdas_unified)
    cv_scores = zeros(nFolds, 1);
    
    for fold = 1:nFolds
        X_fold_train_raw = X_train_raw(cvFolds{fold}.train, :);
        X_fold_val_raw = X_train_raw(cvFolds{fold}.val, :);
        
        X_fold_mean = mean(X_fold_train_raw, 1);
        X_fold_std = std(X_fold_train_raw, 0, 1);
        X_fold_std(X_fold_std < 1e-10) = 1;
        
        X_fold_train = (X_fold_train_raw - X_fold_mean) ./ X_fold_std;
        X_fold_val = (X_fold_val_raw - X_fold_mean) ./ X_fold_std;
        
        B = ridge(y_train(cvFolds{fold}.train), X_fold_train, ridge_lambdas_unified(i), 0);
        y_pred = [ones(sum(cvFolds{fold}.val),1), X_fold_val] * B;
        cv_scores(fold) = sqrt(mean((y_train(cvFolds{fold}.val) - y_pred).^2));
    end
    ridge_cv_results(i) = mean(cv_scores);
end
[ridge_cv_rmse, best_ridge_idx] = min(ridge_cv_results);
best_ridge_lambda = ridge_lambdas_unified(best_ridge_idx);
fprintf('Ridge: lambda=%.4f, CV-RMSE=%.4f\n', best_ridge_lambda, ridge_cv_rmse);

fprintf('\nHyperparameter tuning complete!\n\n');

fprintf('=== Training Final Models ===\n');

rf_optimized = TreeBagger(rf_nTrees, X_train, y_train, 'Method', 'regression', ...
    'OOBPrediction', 'on', 'OOBPredictorImportance', 'on', ...
    'MinLeafSize', best_rf_minLeaf, 'NumPredictorsToSample', best_rf_mtry);

weakLearner = templateTree('MaxNumSplits', 2^best_lsb_depth - 1, ...
    'MinLeafSize', best_lsb_minLeaf);
lsb_optimized = fitrensemble(X_train, y_train, 'Method', 'LSBoost', ...
    'NumLearningCycles', best_lsb_learners, 'Learners', weakLearner, ...
    'LearnRate', best_lsb_learnRate);

[B_lasso, FitInfo_lasso] = lasso(X_train, y_train, 'Lambda', best_lambda_lasso_final);
mdl_lasso_coef = B_lasso;
lasso_intercept = FitInfo_lasso.Intercept;

if all(abs(mdl_lasso_coef) < 1e-10)
    warning('Lasso produced all-zero coefficients! Lambda may be too large.');
end

B_ridge_final = ridge(y_train, X_train, best_ridge_lambda, 0);
ridge_intercept = B_ridge_final(1);
ridge_coef = B_ridge_final(2:end);

fprintf('All models trained successfully!\n');

y_pred_train_opt = predict(rf_optimized, X_train);
y_pred_test_opt  = predict(rf_optimized, X_test);
y_pred_train_lsb = predict(lsb_optimized, X_train);
y_pred_test_lsb  = predict(lsb_optimized, X_test);
y_pred_train_lasso = X_train * mdl_lasso_coef + lasso_intercept;
y_pred_test_lasso  = X_test * mdl_lasso_coef + lasso_intercept;
y_pred_train_ridge = [ones(size(X_train,1),1), X_train] * B_ridge_final;
y_pred_test_ridge  = [ones(size(X_test,1),1), X_test] * B_ridge_final;

rmse_func = @(y, yhat) sqrt(mean((y - yhat).^2));
mae_func  = @(y, yhat) mean(abs(y - yhat));
r2_func   = @(y, yhat) 1 - sum((y - yhat).^2) / sum((y - mean(y)).^2);
mape_func = @(y, yhat) mean(abs((y - yhat) ./ max(abs(y), 1e-10))) * 100;

models = {'OptRF', 'LSBoost', 'Lasso', 'Ridge'};
train_preds = {y_pred_train_opt, y_pred_train_lsb, y_pred_train_lasso, y_pred_train_ridge};
test_preds = {y_pred_test_opt, y_pred_test_lsb, y_pred_test_lasso, y_pred_test_ridge};
all_colors = {modelColors.optrf, modelColors.lsb, modelColors.lasso, modelColors.ridge};

metrics = struct();
for i = 1:4
    metrics.(models{i}).train_rmse = rmse_func(y_train, train_preds{i});
    metrics.(models{i}).test_rmse  = rmse_func(y_test, test_preds{i});
    metrics.(models{i}).train_mae  = mae_func(y_train, train_preds{i});
    metrics.(models{i}).test_mae   = mae_func(y_test, test_preds{i});
    metrics.(models{i}).train_r2   = r2_func(y_train, train_preds{i});
    metrics.(models{i}).test_r2    = r2_func(y_test, test_preds{i});
    metrics.(models{i}).test_mape  = mape_func(y_test, test_preds{i});
end

residuals_opt_train = y_train - y_pred_train_opt;
residuals_opt_test  = y_test - y_pred_test_opt;
residuals_lsb_train = y_train - y_pred_train_lsb;
residuals_lsb_test  = y_test - y_pred_test_lsb;
residuals_lasso_train = y_train - y_pred_train_lasso;
residuals_lasso_test  = y_test - y_pred_test_lasso;
residuals_ridge_train = y_train - y_pred_train_ridge;
residuals_ridge_test  = y_test - y_pred_test_ridge;

residuals_all_train = {residuals_opt_train, residuals_lsb_train, residuals_lasso_train, residuals_ridge_train};
residuals_all_test = {residuals_opt_test, residuals_lsb_test, residuals_lasso_test, residuals_ridge_test};

fprintf('\n=== Statistical Significance Testing ===\n');
fprintf('NOTE: Friedman test on battery-level LOBO results (N=4 batteries)\n');
fprintf('This approach respects independence assumption better than sample-level test\n\n');

lobo_rmse_matrix = zeros(4, 4);
for m = 1:4
    lobo_rmse_matrix(:, m) = lobo_results_all.(models_list{m}).rmse;
end

try
    [p_friedman, tbl_friedman, stats_friedman] = friedman(lobo_rmse_matrix, 1, 'off');
    fprintf('Friedman Test on LOBO RMSE (N=4 batteries): p-value = %.4f\n', p_friedman);
    
    n = size(lobo_rmse_matrix, 1);
    k = size(lobo_rmse_matrix, 2);
    ranks = zeros(n, k);
    for row = 1:n
        [~, order] = sort(lobo_rmse_matrix(row, :));
        ranks(row, order) = 1:k;
    end
    rank_sums = sum(ranks, 1);
    mean_ranks = rank_sums / n;
    
    chi2_stat = (12 * n / (k * (k + 1))) * sum((mean_ranks - (k+1)/2).^2);
    W = chi2_stat / (n * (k - 1));
    
    fprintf('  Chi-square statistic = %.4f\n', chi2_stat);
    fprintf('  Effect Size (Kendall W) = %.4f ', W);
    if W < 0.1
        fprintf('(negligible)\n');
    elseif W < 0.3
        fprintf('(small)\n');
    elseif W < 0.5
        fprintf('(medium)\n');
    else
        fprintf('(large)\n');
    end
    
    fprintf('\n  Mean Ranks (lower is better):\n');
    for i = 1:4
        fprintf('    %s: %.3f\n', model_names_display{i}, mean_ranks(i));
    end
    
    if p_friedman < 0.05
        fprintf('\n  Significant differences detected (p < 0.05)\n');
        
        fprintf('\n  Post-hoc: Nemenyi Test\n');
        q_alpha_table = [0, 1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164];
        
        if k <= 10
            q_alpha = q_alpha_table(k);
            CD = q_alpha * sqrt(k * (k + 1) / (6 * n));
            fprintf('  q_alpha (k=%d, alpha=0.05) = %.3f\n', k, q_alpha);
            fprintf('  Critical Difference (CD) = %.4f\n', CD);
            
            fprintf('\n  Pairwise comparisons:\n');
            comparisons = nchoosek(1:4, 2);
            for c = 1:size(comparisons, 1)
                i = comparisons(c, 1);
                j = comparisons(c, 2);
                rank_diff = abs(mean_ranks(i) - mean_ranks(j));
                
                if rank_diff > CD
                    sig_marker = '** SIGNIFICANT';
                else
                    sig_marker = '';
                end
                
                fprintf('    %s vs %s: |DeltaRank| = %.3f %s\n', ...
                    model_names_display{i}, model_names_display{j}, rank_diff, sig_marker);
            end
        end
    else
        fprintf('\n  No significant differences detected (p >= 0.05)\n');
        fprintf('  NOTE: With only N=4 batteries, statistical power is limited\n');
    end
catch ME
    fprintf('  Statistical tests error: %s\n', ME.message);
end

fprintf('\n=== Residual Normality Tests (All Models) ===\n');

for i = 1:4
    residuals_test = residuals_all_test{i};
    
    try
        [h_lillie, p_lillie] = lillietest(residuals_test);
        
        skewness_val = skewness(residuals_test);
        kurtosis_val = kurtosis(residuals_test);
        
        fprintf('%s:\n', model_names_display{i});
        fprintf('  Lilliefors test: p = %.4f (H0 rejected: %d)\n', p_lillie, h_lillie);
        fprintf('  Skewness: %.4f (ideal: 0)\n', skewness_val);
        fprintf('  Kurtosis: %.4f (ideal: 3)\n', kurtosis_val);
        
        if h_lillie == 0
            fprintf('  Cannot reject normality assumption\n');
        else
            fprintf('  Residuals may not be normally distributed\n');
        end
        fprintf('\n');
    catch ME
        fprintf('%s: Test failed - %s\n\n', model_names_display{i}, ME.message);
    end
end

fprintf('=== Performance Summary ===\n');
fprintf('%-10s | %-12s | %-8s | %-8s | %-8s\n', 'Model', 'Test RMSE', 'R2', 'MAE', 'MAPE(%)');
fprintf('-----------|--------------|----------|----------|----------\n');
for i = 1:4
    fprintf('%-10s | %12.4f | %8.4f | %8.4f | %8.2f\n', model_names_display{i}, ...
        metrics.(models{i}).test_rmse, metrics.(models{i}).test_r2, ...
        metrics.(models{i}).test_mae, metrics.(models{i}).test_mape);
end

fprintf('\n=== Model Stability Assessment (5 Runs) ===\n');
n_stability_runs = 5;
stability_results = zeros(n_stability_runs, 4);

for run = 1:n_stability_runs
    fprintf('  Run %d/%d...', run, n_stability_runs);
    rng(42 + run, 'twister');
    
    rf_temp = TreeBagger(rf_nTrees, X_train, y_train, 'Method', 'regression', ...
        'MinLeafSize', best_rf_minLeaf, 'NumPredictorsToSample', best_rf_mtry);
    stability_results(run, 1) = rmse_func(y_test, predict(rf_temp, X_test));
    
    weakLearner = templateTree('MaxNumSplits', 2^best_lsb_depth - 1, ...
        'MinLeafSize', best_lsb_minLeaf);
    lsb_temp = fitrensemble(X_train, y_train, 'Method', 'LSBoost', ...
        'NumLearningCycles', best_lsb_learners, 'Learners', weakLearner, ...
        'LearnRate', best_lsb_learnRate);
    stability_results(run, 2) = rmse_func(y_test, predict(lsb_temp, X_test));
    
    stability_results(run, 3) = metrics.Lasso.test_rmse;
    stability_results(run, 4) = metrics.Ridge.test_rmse;
    
    fprintf(' Done\n');
end

fprintf('\nStability Results:\n');
fprintf('%-10s | %-12s | %-12s | %-12s\n', 'Model', 'Mean', 'Std', 'CV(%)');
fprintf('-----------|--------------|--------------|-------------\n');
for i = 1:4
    mean_rmse = mean(stability_results(:, i));
    std_rmse = std(stability_results(:, i));
    cv_pct = 100 * std_rmse / (mean_rmse + eps);
    fprintf('%-10s | %12.4f | %12.4f | %12.2f\n', model_names_display{i}, ...
        mean_rmse, std_rmse, cv_pct);
end

fprintf('\n=== Feature Importance Analysis ===\n');
fprintf('Computing unified permutation importance for all models...\n');

n_perm = 30;
perm_importance_rf = zeros(n_features, 1);
perm_importance_lsb = zeros(n_features, 1);
perm_importance_lasso = zeros(n_features, 1);
perm_importance_ridge = zeros(n_features, 1);

baseline_rmse_rf = rmse_func(y_test, y_pred_test_opt);
baseline_rmse_lsb = rmse_func(y_test, y_pred_test_lsb);
baseline_rmse_lasso = rmse_func(y_test, y_pred_test_lasso);
baseline_rmse_ridge = rmse_func(y_test, y_pred_test_ridge);

for feat = 1:n_features
    perm_scores_rf = zeros(n_perm, 1);
    perm_scores_lsb = zeros(n_perm, 1);
    perm_scores_lasso = zeros(n_perm, 1);
    perm_scores_ridge = zeros(n_perm, 1);
    
    for p = 1:n_perm
        X_test_perm = X_test;
        perm_idx = randperm(size(X_test, 1));
        X_test_perm(:, feat) = X_test(perm_idx, feat);
        
        perm_scores_rf(p) = rmse_func(y_test, predict(rf_optimized, X_test_perm));
        perm_scores_lsb(p) = rmse_func(y_test, predict(lsb_optimized, X_test_perm));
        perm_scores_lasso(p) = rmse_func(y_test, X_test_perm * mdl_lasso_coef + lasso_intercept);
        perm_scores_ridge(p) = rmse_func(y_test, [ones(size(X_test_perm,1),1), X_test_perm] * B_ridge_final);
    end
    
    perm_importance_rf(feat) = mean(perm_scores_rf) - baseline_rmse_rf;
    perm_importance_lsb(feat) = mean(perm_scores_lsb) - baseline_rmse_lsb;
    perm_importance_lasso(feat) = mean(perm_scores_lasso) - baseline_rmse_lasso;
    perm_importance_ridge(feat) = mean(perm_scores_ridge) - baseline_rmse_ridge;
end

perm_importance_rf = safe_normalize_vec(perm_importance_rf);
perm_importance_lsb = safe_normalize_vec(perm_importance_lsb);
perm_importance_lasso = safe_normalize_vec(perm_importance_lasso);
perm_importance_ridge = safe_normalize_vec(perm_importance_ridge);

perm_importance_all = {perm_importance_rf, perm_importance_lsb, ...
    perm_importance_lasso, perm_importance_ridge};

% FIXED: 添加Cycle特征重要性警告
fprintf('\n=== Cycle Feature Importance Warning ===\n');
[~, rank_idx_lsb] = sort(perm_importance_lsb, 'descend');
cycle_rank = find(rank_idx_lsb == 1);
fprintf('Cycle feature importance rank in LSBoost: %d/%d\n', cycle_rank, n_features);
if cycle_rank <= 3
    fprintf('WARNING: Cycle is among top-3 features!\n');
    fprintf('This may indicate the model relies heavily on cycle number rather than\n');
    fprintf('learning degradation mechanisms from operational features.\n');
    fprintf('Consider discussing this limitation in the paper or performing ablation study.\n');
end

n_nonzero_lasso = sum(abs(mdl_lasso_coef) > 1e-10);
fprintf('\nLasso Feature Selection: %d/%d non-zero (%.1f%%)\n', ...
    n_nonzero_lasso, length(mdl_lasso_coef), 100 * n_nonzero_lasso / length(mdl_lasso_coef));

rdylbu_colors = [
    0.196, 0.004, 0.965; 0.365, 0.310, 0.937; 0.498, 0.565, 0.973;
    0.670, 0.745, 0.988; 0.878, 0.922, 0.973; 1.000, 1.000, 0.749;
    0.996, 0.878, 0.545; 0.992, 0.682, 0.380; 0.937, 0.396, 0.282;
    0.843, 0.188, 0.153
];

rdbu_colors_shap = [
    0.019, 0.188, 0.380; 0.129, 0.400, 0.675; 0.400, 0.651, 0.808;
    0.698, 0.875, 0.929; 0.969, 0.957, 0.961; 0.992, 0.859, 0.780;
    0.957, 0.643, 0.376; 0.843, 0.188, 0.153; 0.698, 0.094, 0.169
];

pdp_colors = [
    0.196, 0.004, 0.965; 0.365, 0.310, 0.937; 0.498, 0.565, 0.973;
    0.843, 0.188, 0.153; 0.937, 0.396, 0.282; 0.992, 0.682, 0.380
];

fprintf('\n=== Generating Figures (1-16) ===\n');

fprintf('  Generating Figure 1...\n');
figure('Position', [100 100 900 700], 'Color', 'w');
for b = 1:4
    if isfield(allData, batteries{b}) && ~isempty(allData.(batteries{b}).capacity)
        subplot(2, 2, b);
        cycles = allData.(batteries{b}).cycleNum;
        cap    = allData.(batteries{b}).capacity;

        plot(cycles, cap, 'o-', 'Color', batteryColors(b,:), ...
            'MarkerFaceColor', batteryColors(b,:), 'MarkerSize', 4, ...
            'LineWidth', 1.5, 'MarkerEdgeColor', 'none');
        hold on; 
        plot(xlim, [1.4 1.4], '--', 'Color', natureColors.deepRed, 'LineWidth', 1.2);

        if numel(cycles) >= 3
            p = polyfit(cycles, cap, 2);
            xfit = linspace(min(cycles), max(cycles), 100);
            yfit = polyval(p, xfit);
            plot(xfit, yfit, '--', 'Color', batteryColors(b,:)*0.7, 'LineWidth', 1);
        end

        xlabel('Cycle Number', 'FontSize', 10);
        ylabel('Capacity (Ah)', 'FontSize', 10);
        title(sprintf('Battery %s', batteries{b}(end-1:end)), 'FontSize', 11, 'FontWeight', 'bold');
        legend('Measured', 'EOL (1.4 Ah)', 'Trend', 'Location', 'best', 'FontSize', 8, 'Box', 'off');
        ylim([max(1.0, min(cap)-0.1) min(2.2, max(cap)+0.1)]);
        grid on; box on;
    end
end
sgtitle('Figure 1: Battery Capacity Degradation Curves', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('  Generating Figure 2...\n');
figure('Position', [100 100 1000 900], 'Color', 'w');

corr_matrix = corr([X_train, y_train]);
feature_names_with_y = [featureNames, 'Capacity'];

imagesc(corr_matrix);
colormap(rdylbu_colors);
caxis([-1 1]);

[n_corr, m_corr] = size(corr_matrix);
for i = 1:n_corr
    for j = 1:m_corr
        val = corr_matrix(i,j);
        if abs(val) > 0.7
            textColor = 'w';
        else
            textColor = 'k';
        end
        text(j, i, sprintf('%.2f', val), ...
            'HorizontalAlignment', 'center', 'FontSize', 7, ...
            'Color', textColor, 'FontWeight', 'bold');
    end
end

set(gca, 'XTick', 1:length(feature_names_with_y), ...
    'XTickLabel', feature_names_with_y, 'XTickLabelRotation', 45, 'FontSize', 9);
set(gca, 'YTick', 1:length(feature_names_with_y), ...
    'YTickLabel', feature_names_with_y, 'FontSize', 9);
axis square; box on;

cb = colorbar;
cb.Label.String = 'Pearson Correlation';
cb.Label.FontSize = 10;

title('Figure 2: Feature Correlation Matrix', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('  Generating Figure 3...\n');
figure('Position', [100 100 1400 1000], 'Color', 'w');

subplot(3,2,1);
[X_grid, Y_grid] = meshgrid(1:length(rf_mtry_ratios_unified), 1:length(rf_minLeafSizes_unified));
surf(X_grid, Y_grid, rf_cv_results, 'FaceAlpha', 0.8, 'EdgeColor', 'none');
colormap(gca, flipud(rdylbu_colors));
cb = colorbar; cb.Label.String = 'CV-RMSE (Ah)';
xlabel('mtry Ratio Index'); ylabel('MinLeafSize Index'); zlabel('CV-RMSE');
set(gca, 'XTick', 1:length(rf_mtry_ratios_unified), 'XTickLabel', rf_mtry_ratios_unified);
set(gca, 'YTick', 1:length(rf_minLeafSizes_unified), 'YTickLabel', rf_minLeafSizes_unified);
title('Optimized Random Forest', 'FontWeight', 'bold');
grid on; box on; view(45, 30);

subplot(3,2,2);
lsb_slice = squeeze(lsb_cv_results(:, :, best_k, best_m));
imagesc(lsb_slice');
colormap(gca, flipud(rdylbu_colors));
cb = colorbar; cb.Label.String = 'CV-RMSE (Ah)';
set(gca, 'XTick', 1:length(lsb_depths_unified), 'XTickLabel', lsb_depths_unified);
set(gca, 'YTick', 1:length(lsb_learners_unified), 'YTickLabel', lsb_learners_unified);
xlabel('Tree Depth'); ylabel('Num Learners');
title(sprintf('LSBoost (LR=%.2f)', best_lsb_learnRate), 'FontWeight', 'bold');
axis square; box on;

subplot(3,2,3);
plot(log10(lasso_lambdas_unified), lasso_cv_scores_per_lambda, 'o-', ...
    'Color', modelColors.lasso, 'LineWidth', 2.5, 'MarkerSize', 6, ...
    'MarkerFaceColor', modelColors.lasso);
hold on;
xline(log10(best_lambda_lasso_final), '--', 'Color', natureColors.deepRed, 'LineWidth', 2);
xlabel('log_{10}(Lambda)'); ylabel('CV-RMSE (Ah)');
title('Lasso Regularization Path', 'FontWeight', 'bold');
grid on; box on;

subplot(3,2,4);
semilogx(ridge_lambdas_unified, ridge_cv_results, 'o-', 'Color', modelColors.ridge, ...
    'LineWidth', 2.5, 'MarkerSize', 6, 'MarkerFaceColor', modelColors.ridge);
hold on;
xline(best_ridge_lambda, '--', 'Color', natureColors.deepRed, 'LineWidth', 2);
xlabel('Lambda'); ylabel('CV-RMSE (Ah)');
title('Ridge Regularization Path', 'FontWeight', 'bold');
grid on; box on;

subplot(3,2,5);
cv_all = [min_rf_rmse; min_lsb_rmse; lasso_cv_rmse; ridge_cv_rmse];
b = bar(cv_all);
b.FaceColor = 'flat';
b.CData(1,:) = modelColors.optrf;
b.CData(2,:) = modelColors.lsb;
b.CData(3,:) = modelColors.lasso;
b.CData(4,:) = modelColors.ridge;
b.FaceAlpha = 0.85;
set(gca, 'XTickLabel', model_names_display, 'XTickLabelRotation', 45);
ylabel('CV-RMSE (Ah)');
title('CV Results', 'FontWeight', 'bold');
grid on; box on;

subplot(3,2,6);
complexity = [rf_nTrees * 2^5; best_lsb_learners * 2^best_lsb_depth; 1; 1];
scatter(complexity, cv_all, 150, [modelColors.optrf; modelColors.lsb; ...
    modelColors.lasso; modelColors.ridge], 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
set(gca, 'XScale', 'log');
xlabel('Model Complexity'); ylabel('CV-RMSE');
title('Complexity-Performance', 'FontWeight', 'bold');
grid on; box on;

sgtitle('Figure 3: Hyperparameter Optimization', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('  Generating Figure 4...\n');
figure('Position', [100 100 1100 900], 'Color', 'w');

test_rmses = [metrics.OptRF.test_rmse; metrics.LSBoost.test_rmse; ...
    metrics.Lasso.test_rmse; metrics.Ridge.test_rmse];
test_r2s = [metrics.OptRF.test_r2; metrics.LSBoost.test_r2; ...
    metrics.Lasso.test_r2; metrics.Ridge.test_r2];
test_maes = [metrics.OptRF.test_mae; metrics.LSBoost.test_mae; ...
    metrics.Lasso.test_mae; metrics.Ridge.test_mae];
test_mapes = [metrics.OptRF.test_mape; metrics.LSBoost.test_mape; ...
    metrics.Lasso.test_mape; metrics.Ridge.test_mape];

subplot(2,2,1);
b = bar(test_rmses);
b.FaceColor = 'flat';
for i = 1:4, b.CData(i,:) = all_colors{i}; end
b.FaceAlpha = 0.85;
set(gca, 'XTickLabel', model_names_display, 'XTickLabelRotation', 45);
ylabel('Test RMSE (Ah)');
title('RMSE', 'FontWeight', 'bold');
grid on; box on;

subplot(2,2,2);
b = bar(test_r2s);
b.FaceColor = 'flat';
for i = 1:4, b.CData(i,:) = all_colors{i}; end
b.FaceAlpha = 0.85;
set(gca, 'XTickLabel', model_names_display, 'XTickLabelRotation', 45);
ylabel('R2');
title('R2 Score', 'FontWeight', 'bold');
grid on; box on;

subplot(2,2,3);
b = bar(test_maes);
b.FaceColor = 'flat';
for i = 1:4, b.CData(i,:) = all_colors{i}; end
b.FaceAlpha = 0.85;
set(gca, 'XTickLabel', model_names_display, 'XTickLabelRotation', 45);
ylabel('Test MAE (Ah)');
title('MAE', 'FontWeight', 'bold');
grid on; box on;

subplot(2,2,4);
b = bar(test_mapes);
b.FaceColor = 'flat';
for i = 1:4, b.CData(i,:) = all_colors{i}; end
b.FaceAlpha = 0.85;
set(gca, 'XTickLabel', model_names_display, 'XTickLabelRotation', 45);
ylabel('MAPE (%)');
title('MAPE', 'FontWeight', 'bold');
grid on; box on;

sgtitle('Figure 4: Model Performance Comparison', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('  Generating Figure 5...\n');
figure('Position', [100 100 1400 700], 'Color', 'w');

for i = 1:4
    subplot(2,4,i);
    scatter(y_train, train_preds{i}, 35, all_colors{i}, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    plot([min(y_train) max(y_train)], [min(y_train) max(y_train)], '--', ...
        'Color', natureColors.charcoal, 'LineWidth', 1.2);
    xlabel('Actual (Ah)'); ylabel('Predicted (Ah)');
    title(sprintf('%s - Train\nRMSE=%.4f', model_names_display{i}, ...
        metrics.(models{i}).train_rmse));
    axis square; box on; grid on;
    
    subplot(2,4,4+i);
    scatter(y_test, test_preds{i}, 35, all_colors{i}, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    plot([min(y_test) max(y_test)], [min(y_test) max(y_test)], '--', ...
        'Color', natureColors.charcoal, 'LineWidth', 1.2);
    xlabel('Actual (Ah)'); ylabel('Predicted (Ah)');
    title(sprintf('%s - Test\nRMSE=%.4f', model_names_display{i}, ...
        metrics.(models{i}).test_rmse));
    axis square; box on; grid on;
end

sgtitle('Figure 5: Prediction Accuracy', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('  Generating Figure 6...\n');
figure('Position', [100 100 1400 700], 'Color', 'w');

for i = 1:4
    subplot(2,4,i);
    scatter(train_preds{i}, residuals_all_train{i}, 35, all_colors{i}, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on; yline(0, '--', 'Color', natureColors.charcoal, 'LineWidth', 1.2);
    xlabel('Predicted (Ah)'); ylabel('Residuals (Ah)');
    title(sprintf('%s - Train', model_names_display{i}));
    box on; grid on;
    
    subplot(2,4,4+i);
    scatter(test_preds{i}, residuals_all_test{i}, 35, all_colors{i}, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on; yline(0, '--', 'Color', natureColors.charcoal, 'LineWidth', 1.2);
    xlabel('Predicted (Ah)'); ylabel('Residuals (Ah)');
    title(sprintf('%s - Test', model_names_display{i}));
    box on; grid on;
end

sgtitle('Figure 6: Residual Analysis', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('  Generating Figure 7...\n');
figure('Position', [100 100 1400 700], 'Color', 'w');

for i = 1:4
    subplot(2,4,i);
    histogram(residuals_all_train{i}, 25, 'FaceColor', all_colors{i}, ...
        'FaceAlpha', 0.75, 'EdgeColor', 'w');
    hold on; xline(0, '--', 'Color', natureColors.charcoal, 'LineWidth', 1.2);
    xlabel('Residuals (Ah)'); ylabel('Frequency');
    title(sprintf('%s - Train', model_names_display{i}));
    box on; grid on;
    
    subplot(2,4,4+i);
    histogram(residuals_all_test{i}, 15, 'FaceColor', all_colors{i}, ...
        'FaceAlpha', 0.75, 'EdgeColor', 'w');
    hold on; xline(0, '--', 'Color', natureColors.charcoal, 'LineWidth', 1.2);
    xlabel('Residuals (Ah)'); ylabel('Frequency');
    title(sprintf('%s - Test', model_names_display{i}));
    box on; grid on;
end

sgtitle('Figure 7: Error Distribution', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('  Generating Figure 8...\n');
figure('Position', [100 100 1400 900], 'Color', 'w');

fig8_colors = [0.90, 0.16, 0.22; 0.00, 0.60, 0.80];

for b = 1:4
    if isfield(allData, batteries{b}) && ~isempty(allData.(batteries{b}).capacity)
        subplot(2,2,b);
        
        battData = allData.(batteries{b});
        X_batt = (battData.features - X_mean) ./ X_std;
        y_actual = battData.capacity(:);
        cycles = battData.cycleNum;
        y_pred = predict(lsb_optimized, X_batt);
        
        plot(cycles, y_actual, 'o-', 'Color', fig8_colors(1,:), 'LineWidth', 2.5, ...
            'MarkerSize', 7, 'MarkerFaceColor', fig8_colors(1,:));
        hold on;
        plot(cycles, y_pred, 's--', 'Color', fig8_colors(2,:), 'LineWidth', 2.5, ...
            'MarkerSize', 7, 'MarkerFaceColor', fig8_colors(2,:));
        yline(1.4, '-.', 'Color', [0.8, 0.2, 0.2], 'LineWidth', 2);
        
        rmse_val = sqrt(mean((y_actual - y_pred).^2));
        r2_val = 1 - sum((y_actual - y_pred).^2) / sum((y_actual - mean(y_actual)).^2);
        
        if b == testBatteryID
            data_type = 'TEST';
        else
            data_type = 'TRAIN';
        end
        
        xlabel('Cycle Number'); ylabel('Capacity (Ah)');
        title(sprintf('Battery %s [%s]\nRMSE=%.4f, R2=%.3f', ...
            batteries{b}(end-1:end), data_type, rmse_val, r2_val), 'FontWeight', 'bold');
        legend('Actual', 'LSBoost', 'EOL', 'Box', 'off');
        grid on; box on;
    end
end

sgtitle('Figure 8: Battery-Specific Forecasting (TRAIN/TEST labeled)', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('  Generating Figure 9...\n');
figure('Position', [100 100 1400 900], 'Color', 'w');

for model_idx = 1:4
    subplot(2, 2, model_idx);
    
    importance = perm_importance_all{model_idx};
    [sorted_importance, sort_idx] = sort(importance, 'ascend');
    sorted_features = featureNames(sort_idx);
    
    barh(sorted_importance, 'FaceColor', all_colors{model_idx}, ...
        'FaceAlpha', 0.85, 'EdgeColor', 'w');
    
    set(gca, 'YTick', 1:length(sorted_features), 'YTickLabel', sorted_features);
    xlabel('Permutation Importance (Normalized)', 'FontWeight', 'bold');
    ylabel('Features', 'FontWeight', 'bold');
    title(sprintf('%s', model_names_display{model_idx}), 'FontWeight', 'bold');
    grid on; box on;
end

sgtitle('Figure 9: Feature Importance (Unified Permutation Method)', 'FontSize', 13, 'FontWeight', 'bold');

% FIXED: Figure 10 - 修正变量名和标题，不再称为SHAP
fprintf('  Generating Figure 10...\n');
figure('Position', [100 100 1400 900], 'Color', 'w');

nature_colors_violin = [
    0.8353, 0.2431, 0.3098;
    0.9725, 0.6706, 0.3804;
    0.4784, 0.6784, 0.8039;
    0.2000, 0.2627, 0.5490;
    0.8980, 0.5882, 0.5765;
    0.9882, 0.8431, 0.6863;
    0.7373, 0.8353, 0.9059;
    0.6000, 0.6000, 0.7843;
    0.1333, 0.5451, 0.1333;
    0.8039, 0.5216, 0.2471;
    0.2549, 0.4118, 0.8824;
    0.7059, 0.3451, 0.0235;
    0.0000, 0.5020, 0.5020;
    0.5020, 0.0000, 0.5020;
];

[~, unified_sort_idx] = sort(perm_importance_lsb, 'descend');
unified_sorted_features = featureNames(unified_sort_idx);

for model_idx = 1:4
    subplot(2, 2, model_idx);
    
    importance = perm_importance_all{model_idx};
    sorted_importance = importance(unified_sort_idx);
    n_features_plot = length(unified_sort_idx);
    
    % FIXED: 重命名变量 - 这不是SHAP值，而是重要性加权特征值
    weighted_feature_values = zeros(size(X_train, 1), n_features_plot);
    for ii = 1:n_features_plot
        feat_idx = unified_sort_idx(ii);
        weighted_feature_values(:, ii) = X_train(:, feat_idx) * sorted_importance(ii);
    end
    
    hold on;
    
    for ii = 1:n_features_plot
        current_data = weighted_feature_values(:, ii);
        
        [density, value] = ksdensity(current_data, 'NumPoints', 100);
        
        max_density = max(density);
        if max_density > 0
            density = density / max_density * 0.4;
        else
            density = zeros(size(density));
        end
        
        color_idx = mod(ii-1, size(nature_colors_violin, 1)) + 1;
        base_color = nature_colors_violin(color_idx, :);
        
        n_segments = 50;
        value_segments = linspace(min(value), max(value), n_segments);
        
        for seg = 1:(n_segments-1)
            seg_mask = value >= value_segments(seg) & value < value_segments(seg+1);
            if sum(seg_mask) > 1
                seg_values = value(seg_mask);
                seg_density = density(seg_mask);
                
                gradient_factor = (seg / n_segments);
                
                light_color = base_color * 0.3 + [0.7, 0.7, 0.7];
                current_color = (1 - gradient_factor) * light_color + gradient_factor * base_color;
                
                patch([ii - seg_density, fliplr(ii * ones(1, length(seg_density)))], ...
                      [seg_values, fliplr(seg_values)], ...
                      current_color, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
                
                patch([ii * ones(1, length(seg_density)), fliplr(ii + seg_density)], ...
                      [seg_values, fliplr(seg_values)], ...
                      current_color, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
            end
        end
        
        plot(ii - density, value, '-', 'Color', base_color * 0.6, 'LineWidth', 1.5);
        plot(ii + density, value, '-', 'Color', base_color * 0.6, 'LineWidth', 1.5);
    end
    
    plot([0.5, n_features_plot + 0.5], [0, 0], '--', ...
         'Color', [0.3 0.3 0.3], 'LineWidth', 2);
    
    xlim([0.5, n_features_plot + 0.5]);
    set(gca, 'XTick', 1:n_features_plot, 'XTickLabel', unified_sorted_features);
    
    xlabel('Features (Ranked by Importance)', 'FontWeight', 'bold', 'FontSize', 10);
    ylabel('Weighted Feature Value', 'FontWeight', 'bold', 'FontSize', 10);  % FIXED: 修正Y轴标签
    title(sprintf('%s', model_names_display{model_idx}), 'FontSize', 11, 'FontWeight', 'bold');
    
    grid on;
    set(gca, 'Box', 'on', 'TickDir', 'out', 'GridAlpha', 0.15, ...
        'Layer', 'top', 'LineWidth', 0.8);
    
    hold off;
end

% FIXED: 修正Figure 10标题 - 不再称为SHAP
sgtitle('Figure 10: Importance-Weighted Feature Distribution', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('  Generating Figure 11...\n');
figure('Position', [100 100 1400 1000], 'Color', 'w');

[~, top_feat_idx] = sort(perm_importance_lsb, 'descend');
top_6_features = top_feat_idx(1:6);

for plot_idx = 1:6
    feat_idx = top_6_features(plot_idx);
    feat_values = linspace(min(X_train(:, feat_idx)), max(X_train(:, feat_idx)), 50);
    pdp_values = zeros(length(feat_values), 1);
    
    for ii = 1:length(feat_values)
        X_temp = X_train;
        X_temp(:, feat_idx) = feat_values(ii);
        pdp_values(ii) = mean(predict(lsb_optimized, X_temp));
    end
    
    subplot(3,3,plot_idx);
    fill([feat_values, fliplr(feat_values)], ...
         [pdp_values', ones(1,length(feat_values))*min(pdp_values)], ...
         pdp_colors(plot_idx,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    hold on;
    plot(feat_values, pdp_values, '-', 'Color', pdp_colors(plot_idx,:), 'LineWidth', 2.5);
    
    xlabel(sprintf('%s', featureNames{feat_idx}), 'FontWeight', 'bold');
    ylabel('Predicted Capacity');
    title(sprintf('%s (Rank %d)', featureNames{feat_idx}, plot_idx), 'FontWeight', 'bold');
    grid on; box on;
end

top_4_features = top_feat_idx(1:4);
pair_combinations = [1,2; 1,3; 2,3];

for pair_idx = 1:3
    feat_1 = top_4_features(pair_combinations(pair_idx,1));
    feat_2 = top_4_features(pair_combinations(pair_idx,2));
    
    n_grid = 20;
    feat_1_values = linspace(min(X_train(:, feat_1)), max(X_train(:, feat_1)), n_grid);
    feat_2_values = linspace(min(X_train(:, feat_2)), max(X_train(:, feat_2)), n_grid);
    
    [F1_grid, F2_grid] = meshgrid(feat_1_values, feat_2_values);
    pdp_2d = zeros(n_grid, n_grid);
    
    for ii = 1:n_grid
        for jj = 1:n_grid
            X_temp = X_train;
            X_temp(:, feat_1) = F1_grid(ii, jj);
            X_temp(:, feat_2) = F2_grid(ii, jj);
            pdp_2d(ii, jj) = mean(predict(lsb_optimized, X_temp));
        end
    end
    
    subplot(3,3,6+pair_idx);
    contourf(F1_grid, F2_grid, pdp_2d, 15, 'LineColor', 'none');
    hold on;
    contour(F1_grid, F2_grid, pdp_2d, 6, 'LineColor', 'k', 'LineWidth', 0.5);
    
    colormap(gca, flipud(rdylbu_colors));
    cb = colorbar; cb.Label.String = 'Capacity';
    
    xlabel(featureNames{feat_1}, 'FontWeight', 'bold');
    ylabel(featureNames{feat_2}, 'FontWeight', 'bold');
    title(sprintf('%s x %s', featureNames{feat_1}, featureNames{feat_2}), 'FontWeight', 'bold');
    box on;
end

sgtitle('Figure 11: Partial Dependence Plots (LSBoost)', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('  Generating Figure 12...\n');
figure('Position', [100 100 1400 900], 'Color', 'w');

if isfield(allData, batteries{testBatteryID}) && ~isempty(allData.(batteries{testBatteryID}).capacity)
    cycles_test = allData.(batteries{testBatteryID}).cycleNum;
    
    subplot(2,3,1);
    for ii = 1:4
        errors = abs(y_test - test_preds{ii});
        plot(cycles_test, errors, 'o-', 'Color', all_colors{ii}, ...
            'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', all_colors{ii}, ...
            'DisplayName', model_names_display{ii});
        hold on;
    end
    xlabel('Cycle Number'); ylabel('Absolute Error (Ah)');
    title('Error Evolution', 'FontWeight', 'bold');
    legend('Location', 'best', 'Box', 'off');
    grid on; box on;
    
    subplot(2,3,2);
    for ii = 1:4
        errors = abs(y_test - test_preds{ii});
        cum_errors = cumsum(errors);
        plot(cycles_test, cum_errors, '-', 'Color', all_colors{ii}, ...
            'LineWidth', 2.5, 'DisplayName', model_names_display{ii});
        hold on;
    end
    xlabel('Cycle Number'); ylabel('Cumulative Error (Ah)');
    title('Cumulative Error', 'FontWeight', 'bold');
    legend('Location', 'best', 'Box', 'off');
    grid on; box on;
    
    subplot(2,3,3);
    for ii = 1:4
        pct_errors = abs((y_test - test_preds{ii}) ./ max(abs(y_test), 1e-10)) * 100;
        plot(cycles_test, pct_errors, 'o-', 'Color', all_colors{ii}, ...
            'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', all_colors{ii}, ...
            'DisplayName', model_names_display{ii});
        hold on;
    end
    xlabel('Cycle Number'); ylabel('Absolute % Error');
    title('Percentage Error', 'FontWeight', 'bold');
    legend('Location', 'best', 'Box', 'off');
    grid on; box on;
    
    subplot(2,3,4);
    window = min(5, floor(length(y_test)/3));
    for ii = 1:4
        errors = abs(y_test - test_preds{ii});
        rolling_mean = movmean(errors, window);
        plot(cycles_test, rolling_mean, '-', 'Color', all_colors{ii}, ...
            'LineWidth', 2.5, 'DisplayName', model_names_display{ii});
        hold on;
    end
    xlabel('Cycle Number'); ylabel('Rolling Mean Error');
    title(sprintf('Rolling Mean (Window=%d)', window), 'FontWeight', 'bold');
    legend('Location', 'best', 'Box', 'off');
    grid on; box on;
    
    subplot(2,3,5);
    for ii = 1:4
        errors = abs(y_test - test_preds{ii});
        error_velocity = [0; diff(errors)];
        plot(cycles_test, error_velocity, 'o-', 'Color', all_colors{ii}, ...
            'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', all_colors{ii}, ...
            'DisplayName', model_names_display{ii});
        hold on;
    end
    yline(0, '--k', 'LineWidth', 1);
    xlabel('Cycle Number'); ylabel('Error Velocity');
    title('Error Rate of Change', 'FontWeight', 'bold');
    legend('Location', 'best', 'Box', 'off');
    grid on; box on;
    
    subplot(2,3,6);
    model_preds_mat = [y_pred_test_opt, y_pred_test_lsb, y_pred_test_lasso, y_pred_test_ridge];
    pred_std = std(model_preds_mat, 0, 2);
    
    plot(cycles_test, pred_std, 'o-', 'Color', natureColors.purple, ...
        'LineWidth', 2.5, 'MarkerSize', 6, 'MarkerFaceColor', natureColors.purple);
    xlabel('Cycle Number'); ylabel('Prediction Std Dev (Ah)');
    title('Model Disagreement (Ensemble Std)', 'FontWeight', 'bold');
    grid on; box on;
end

sgtitle('Figure 12: Error Evolution Analysis', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('  Generating Figure 13...\n');
figure('Position', [100 100 1400 900], 'Color', 'w');

subplot(2,3,1);
for ii = 1:4
    [f, x] = ecdf(abs(y_test - test_preds{ii}));
    plot(x, f, 'LineWidth', 2.5, 'Color', all_colors{ii}, ...
        'DisplayName', model_names_display{ii});
    hold on;
end
xlabel('Absolute Error (Ah)'); ylabel('Cumulative Probability');
title('Error CDF', 'FontWeight', 'bold');
legend('Box', 'off');
grid on; box on;

subplot(2,3,2);
boxData = [abs(y_test - y_pred_test_opt), abs(y_test - y_pred_test_lsb), ...
           abs(y_test - y_pred_test_lasso), abs(y_test - y_pred_test_ridge)];
h = boxplot(boxData, 'Labels', model_names_display, 'Colors', 'k');
set(h, 'LineWidth', 1.5);
set(gca, 'XTickLabelRotation', 45);
ylabel('Absolute Error (Ah)');
title('Error Distribution', 'FontWeight', 'bold');
hold on;
bp = findobj(gca, 'Tag', 'Box');
for ii = 1:length(bp)
    patch(get(bp(ii), 'XData'), get(bp(ii), 'YData'), all_colors{5-ii}, 'FaceAlpha', 0.6);
end
grid on; box on;

subplot(2,3,3);
max_lag = min(20, floor(length(y_test)/4));
if max_lag >= 1
    colors_acf = {modelColors.optrf, modelColors.lsb, modelColors.lasso, modelColors.ridge};
    hold on;
    
    for ii = 1:4
        try
            [acf_vals, lags] = autocorr(residuals_all_test{ii}, 'NumLags', max_lag);
        catch
            [acf_vals, lags] = autocorr(residuals_all_test{ii}, max_lag);
        end
        
        line_styles = {'-o', '-s', '-^', '-d'};
        plot(lags, acf_vals, line_styles{ii}, 'Color', colors_acf{ii}, 'LineWidth', 1.5, ...
            'MarkerSize', 4, 'MarkerFaceColor', colors_acf{ii}, ...
            'DisplayName', model_names_display{ii});
    end
    
    conf_bound = 1.96/sqrt(length(y_test));
    yline(conf_bound, '--r', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    yline(-conf_bound, '--r', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    xlabel('Lag'); ylabel('Autocorrelation');
    title('Residual Autocorrelation', 'FontWeight', 'bold');
    legend('Location', 'best', 'Box', 'off');
    ylim([-0.5 1]);
    grid on; box on;
end

subplot(2,3,4);
for ii = 1:4
    residuals_sorted = sort(residuals_all_test{ii});
    n_pts = length(residuals_sorted);
    p = ((1:n_pts) - 0.5) / n_pts;
    theoretical_quantiles = norminv(p);
    
    plot(theoretical_quantiles, residuals_sorted, 'o', 'Color', all_colors{ii}, ...
        'MarkerSize', 4, 'MarkerFaceColor', all_colors{ii}, ...
        'DisplayName', model_names_display{ii});
    hold on;
    
    mu_i = mean(residuals_all_test{ii});
    sigma_i = std(residuals_all_test{ii});
    xl = xlim;
    plot(xl, xl * sigma_i + mu_i, '-', 'Color', all_colors{ii} * 0.6, ...
        'LineWidth', 1, 'HandleVisibility', 'off');
end
xlabel('Theoretical Quantiles'); ylabel('Sample Quantiles');
title('Q-Q Plot (All Models)', 'FontWeight', 'bold');
legend('Location', 'best', 'Box', 'off');
grid on; box on;

subplot(2,3,5);
capacity_bins = linspace(min(y_test), max(y_test), 5);
bin_errors = zeros(length(capacity_bins)-1, 4);
for ii = 1:length(capacity_bins)-1
    idx = y_test >= capacity_bins(ii) & y_test < capacity_bins(ii+1);
    if sum(idx) > 0
        for jj = 1:4
            bin_errors(ii,jj) = rmse_func(y_test(idx), test_preds{jj}(idx));
        end
    end
end
b = bar(bin_errors);
for ii = 1:4
    b(ii).FaceColor = all_colors{ii};
    b(ii).FaceAlpha = 0.85;
end
bin_labels = arrayfun(@(x,y) sprintf('%.2f-%.2f', x, y), ...
    capacity_bins(1:end-1), capacity_bins(2:end), 'UniformOutput', false);
set(gca, 'XTickLabel', bin_labels, 'XTickLabelRotation', 45);
ylabel('RMSE (Ah)');
title('Error by Capacity Range', 'FontWeight', 'bold');
legend(model_names_display, 'Box', 'off');
grid on; box on;

subplot(2,3,6);
ape_all = [abs((y_test - y_pred_test_opt)./max(abs(y_test),1e-10))*100, ...
           abs((y_test - y_pred_test_lsb)./max(abs(y_test),1e-10))*100, ...
           abs((y_test - y_pred_test_lasso)./max(abs(y_test),1e-10))*100, ...
           abs((y_test - y_pred_test_ridge)./max(abs(y_test),1e-10))*100];
h = boxplot(ape_all, 'Labels', model_names_display, 'Colors', 'k');
set(h, 'LineWidth', 1.5);
set(gca, 'XTickLabelRotation', 45);
ylabel('Absolute % Error');
title('Percentage Error', 'FontWeight', 'bold');
hold on;
bp = findobj(gca, 'Tag', 'Box');
for ii = 1:length(bp)
    patch(get(bp(ii), 'XData'), get(bp(ii), 'YData'), all_colors{5-ii}, 'FaceAlpha', 0.6);
end
grid on; box on;

sgtitle('Figure 13: Statistical Error Analysis', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('  Generating Figure 14...\n');
figure('Position', [100 100 1400 700], 'Color', 'w');

subplot(2,3,1);
plot(oobError(rf_optimized), 'Color', modelColors.optrf, 'LineWidth', 2);
xlabel('Number of Trees'); ylabel('OOB MSE');
title('Opt RF - OOB Error', 'FontWeight', 'bold');
grid on; box on;

subplot(2,3,[2,3]);

fprintf('  Computing learning curves (stratified early-cycle sampling)...\n');
train_fractions = [0.2, 0.4, 0.6, 0.8, 1.0];
train_sizes = round(train_fractions * sum(trainIdx));
lc_train_rmse = zeros(length(train_sizes), 4);
lc_val_rmse = zeros(length(train_sizes), 4);

unique_train_batts = unique(trainBattID);

for ii = 1:length(train_sizes)
    target_n = train_sizes(ii);
    
    idx_sample = [];
    n_per_batt = floor(target_n / length(unique_train_batts));
    
    for batt = unique_train_batts'
        batt_mask = find(trainBattID == batt);
        batt_cycles = trainCycleID(batt_mask);
        [~, cycle_order] = sort(batt_cycles);
        sorted_batt_idx = batt_mask(cycle_order);
        n_take = min(n_per_batt, length(sorted_batt_idx));
        idx_sample = [idx_sample; sorted_batt_idx(1:n_take)];
    end
    
    if length(idx_sample) < target_n
        remaining = target_n - length(idx_sample);
        all_remaining_idx = setdiff(1:length(trainBattID), idx_sample);
        if ~isempty(all_remaining_idx) && remaining > 0
            add_idx = all_remaining_idx(1:min(remaining, length(all_remaining_idx)));
            idx_sample = [idx_sample; add_idx(:)];
        end
    end
    
    X_sample_raw = X_train_raw(idx_sample, :);
    y_sample = y_train(idx_sample);
    
    X_sample_mean = mean(X_sample_raw, 1);
    X_sample_std = std(X_sample_raw, 0, 1);
    X_sample_std(X_sample_std < 1e-10) = 1;
    
    X_sample = (X_sample_raw - X_sample_mean) ./ X_sample_std;
    X_test_scaled = (X_test_raw - X_sample_mean) ./ X_sample_std;
    
    rf_temp = TreeBagger(rf_nTrees, X_sample, y_sample, ...
        'Method', 'regression', 'MinLeafSize', best_rf_minLeaf, ...
        'NumPredictorsToSample', best_rf_mtry);
    lc_train_rmse(ii,1) = rmse_func(y_sample, predict(rf_temp, X_sample));
    lc_val_rmse(ii,1) = rmse_func(y_test, predict(rf_temp, X_test_scaled));
    
    weakLearner = templateTree('MaxNumSplits', 2^best_lsb_depth - 1, ...
        'MinLeafSize', best_lsb_minLeaf);
    lsb_temp = fitrensemble(X_sample, y_sample, ...
        'Method', 'LSBoost', 'NumLearningCycles', best_lsb_learners, ...
        'Learners', weakLearner, 'LearnRate', best_lsb_learnRate);
    lc_train_rmse(ii,2) = rmse_func(y_sample, predict(lsb_temp, X_sample));
    lc_val_rmse(ii,2) = rmse_func(y_test, predict(lsb_temp, X_test_scaled));
    
    [B_temp, FitInfo_temp] = lasso(X_sample, y_sample, 'Lambda', best_lambda_lasso_final);
    y_pred_train_lc = X_sample * B_temp + FitInfo_temp.Intercept;
    y_pred_test_lc = X_test_scaled * B_temp + FitInfo_temp.Intercept;
    lc_train_rmse(ii,3) = rmse_func(y_sample, y_pred_train_lc);
    lc_val_rmse(ii,3) = rmse_func(y_test, y_pred_test_lc);
    
    B_temp = ridge(y_sample, X_sample, best_ridge_lambda, 0);
    y_pred_train_lc = [ones(length(y_sample),1), X_sample] * B_temp;
    y_pred_test_lc = [ones(size(X_test_scaled,1),1), X_test_scaled] * B_temp;
    lc_train_rmse(ii,4) = rmse_func(y_sample, y_pred_train_lc);
    lc_val_rmse(ii,4) = rmse_func(y_test, y_pred_test_lc);
end

for ii = 1:4
    plot(train_sizes, lc_train_rmse(:,ii), 'o-', 'Color', all_colors{ii}, ...
        'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', [model_names_display{ii} ' Train']);
    hold on;
    plot(train_sizes, lc_val_rmse(:,ii), 's--', 'Color', all_colors{ii}, ...
        'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', [model_names_display{ii} ' Test']);
end
xlabel('Training Set Size'); ylabel('RMSE (Ah)');

title('Learning Curves (Stratified Early-Cycle Sampling)', 'FontWeight', 'bold');
legend('Box', 'off', 'Location', 'best', 'NumColumns', 2);
grid on; box on;

subplot(2,3,4);
data_fractions = train_sizes / max(train_sizes);
for ii = 1:4
    plot(data_fractions, lc_val_rmse(:,ii), '-', 'Color', all_colors{ii}, ...
        'LineWidth', 2.5, 'DisplayName', model_names_display{ii});
    hold on;
end
xlabel('Fraction of Training Data'); ylabel('Test RMSE');
title('Data Efficiency', 'FontWeight', 'bold');
legend('Box', 'off');
grid on; box on;

subplot(2,3,5);
gen_gap = lc_val_rmse - lc_train_rmse;
for ii = 1:4
    plot(train_sizes, gen_gap(:,ii), 'o-', 'Color', all_colors{ii}, ...
        'LineWidth', 2, 'MarkerSize', 6, 'DisplayName', model_names_display{ii});
    hold on;
end
xlabel('Training Set Size'); ylabel('Test-Train RMSE Gap');
title('Generalization Gap', 'FontWeight', 'bold');
legend('Box', 'off');
grid on; box on;

subplot(2,3,6);
final_perf = [lc_val_rmse(end,:); lc_train_rmse(end,:)]';
b = bar(final_perf, 'grouped');
b(1).FaceColor = natureColors.deepRed; b(1).FaceAlpha = 0.85;
b(2).FaceColor = natureColors.green; b(2).FaceAlpha = 0.85;
set(gca, 'XTickLabel', model_names_display, 'XTickLabelRotation', 45);
ylabel('RMSE (Ah)');
title('Final Performance', 'FontWeight', 'bold');
legend('Test', 'Train', 'Box', 'off');
grid on; box on;

sgtitle('Figure 14: Learning Curve Analysis', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('  Generating Figure 15...\n');
figure('Position', [100 100 1400 900], 'Color', 'w');

subplot(2,3,1);
for b = 1:3
    if isfield(allData, batteries{b}) && numel(allData.(batteries{b}).capacity) >= 2
        cap = allData.(batteries{b}).capacity(:);
        cycles = allData.(batteries{b}).cycleNum(:);
        
        cycle_diff = diff(cycles);
        cap_diff = -diff(cap);
        valid_idx = (cycle_diff > 0) & isfinite(cap_diff);
        
        deg_rate = nan(size(cycle_diff));
        deg_rate(valid_idx) = cap_diff(valid_idx) ./ cycle_diff(valid_idx);
        
        plot(cycles(2:end), deg_rate, 'o-', 'Color', batteryColors(b,:), ...
            'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerFaceColor', batteryColors(b,:));
        hold on;
    end
end
xlabel('Cycle'); ylabel('Degradation Rate (Ah/cycle)');
title('Capacity Degradation Rate', 'FontWeight', 'bold');
legend('B05','B06','B07', 'Box', 'off');
grid on; box on;

subplot(2,3,[2,3]);
if isfield(allData, batteries{testBatteryID})
    battData = allData.(batteries{testBatteryID});
    y_actual = battData.capacity(:);
    cycles = battData.cycleNum(:);
    
    y_pred_mean = y_pred_test_lsb;
    model_preds_mat = [y_pred_test_opt, y_pred_test_lsb, y_pred_test_lasso, y_pred_test_ridge];
    pred_std = std(model_preds_mat, 0, 2);
    pred_lower = y_pred_mean - 2 * pred_std;
    pred_upper = y_pred_mean + 2 * pred_std;
    
    plot(cycles, y_actual, 'o', 'Color', natureColors.charcoal, ...
        'MarkerSize', 7, 'MarkerFaceColor', natureColors.charcoal);
    hold on;
    plot(cycles, y_pred_mean, '-', 'Color', modelColors.lsb, 'LineWidth', 2.5);
    fill([cycles; flipud(cycles)], [pred_upper; flipud(pred_lower)], ...
        modelColors.lsb, 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    
    xlabel('Cycle'); ylabel('Capacity (Ah)');
    title('Prediction with Model Disagreement Band (+/-2 Std)', 'FontWeight', 'bold');
    legend('Actual', 'LSBoost Mean', 'Model Disagreement (+/-2 Std)', 'Box', 'off');
    grid on; box on;
end

subplot(2,3,4);
model_preds_matrix = [y_pred_test_opt, y_pred_test_lsb, y_pred_test_lasso, y_pred_test_ridge];
pred_std = std(model_preds_matrix, 0, 2);
scatter(y_test, pred_std, 60, pred_std, 'filled', 'MarkerFaceAlpha', 0.7);
colormap(gca, flipud(rdylbu_colors));
cb = colorbar; cb.Label.String = 'Std Dev (Ah)';
xlabel('Actual Capacity (Ah)'); ylabel('Model Disagreement (Std)');
title('Ensemble Disagreement', 'FontWeight', 'bold');
grid on; box on;

subplot(2,3,5);
capacity_corr = corr_matrix(end, 1:end-1);
[~, sort_idx] = sort(abs(capacity_corr), 'descend');
top_10 = sort_idx(1:min(10, length(sort_idx)));
b = barh(capacity_corr(top_10));
b.FaceColor = 'flat';
for ii = 1:length(top_10)
    if capacity_corr(top_10(ii)) > 0
        b.CData(ii,:) = natureColors.deepBlue;
    else
        b.CData(ii,:) = natureColors.deepRed;
    end
end
b.FaceAlpha = 0.85;
set(gca, 'YTick', 1:length(top_10), 'YTickLabel', featureNames(top_10));
xlabel('Pearson Correlation');
title('Top Feature Correlations', 'FontWeight', 'bold');
xline(0, '--k');
grid on; box on;

subplot(2,3,6);
train_test_gap = [abs(metrics.OptRF.train_rmse - metrics.OptRF.test_rmse);
                  abs(metrics.LSBoost.train_rmse - metrics.LSBoost.test_rmse);
                  abs(metrics.Lasso.train_rmse - metrics.Lasso.test_rmse);
                  abs(metrics.Ridge.train_rmse - metrics.Ridge.test_rmse)];
b = bar(train_test_gap);
b.FaceColor = 'flat';
for ii = 1:4, b.CData(ii,:) = all_colors{ii}; end
b.FaceAlpha = 0.85;
set(gca, 'XTickLabel', model_names_display, 'XTickLabelRotation', 45);
ylabel('|Train-Test| RMSE (Ah)');
title('Generalization Gap', 'FontWeight', 'bold');
grid on; box on;

sgtitle('Figure 15: Advanced Model Diagnostics', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('  Generating Figure 16...\n');
figure('Position', [100 100 1200 850], 'Color', 'w');

subplot(2,2,1);
metrics_matrix = [test_rmses'; test_r2s'; test_maes'; test_mapes'/10];
b = bar(metrics_matrix, 'grouped');
for ii = 1:4
    b(ii).FaceColor = all_colors{ii};
    b(ii).FaceAlpha = 0.85;
end
set(gca, 'XTickLabel', {'RMSE', 'R2', 'MAE', 'MAPE/10'});
ylabel('Normalized Value');
title('Metrics Comparison', 'FontWeight', 'bold');
legend(model_names_display, 'Box', 'off', 'Location', 'best');
grid on; box on;

subplot(2,2,2);
scores = zeros(4, 5);

range_rmse = max(test_rmses) - min(test_rmses);
if range_rmse > 1e-10
    scores(:,1) = (max(test_rmses) - test_rmses) / range_rmse;
else
    scores(:,1) = 1;
end

range_r2 = max(test_r2s) - min(test_r2s);
if range_r2 > 1e-10
    scores(:,2) = (test_r2s - min(test_r2s)) / range_r2;
else
    scores(:,2) = 1;
end

range_mae = max(test_maes) - min(test_maes);
if range_mae > 1e-10
    scores(:,3) = (max(test_maes) - test_maes) / range_mae;
else
    scores(:,3) = 1;
end

range_mape = max(test_mapes) - min(test_mapes);
if range_mape > 1e-10
    scores(:,4) = (max(test_mapes) - test_mapes) / range_mape;
else
    scores(:,4) = 1;
end

range_gap = max(train_test_gap) - min(train_test_gap);
if range_gap > 1e-10
    scores(:,5) = (max(train_test_gap) - train_test_gap) / range_gap;
else
    scores(:,5) = 1;
end

overall_scores = mean(scores, 2);
overall_scores_pct = overall_scores * 100;

b = bar(overall_scores_pct);
b.FaceColor = 'flat';
for ii = 1:4, b.CData(ii,:) = all_colors{ii}; end
b.FaceAlpha = 0.85;
set(gca, 'XTickLabel', model_names_display, 'XTickLabelRotation', 45);
ylabel('Overall Score (0-100)');
title('Overall Performance Score', 'FontWeight', 'bold');
ylim([0 105]);
for ii = 1:4
    text(ii, overall_scores_pct(ii)+2, sprintf('%.1f', overall_scores_pct(ii)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end
grid on; box on;

subplot(2,2,3);
consensus_imp = (perm_importance_rf + perm_importance_lsb + ...
                 perm_importance_lasso + perm_importance_ridge) / 4;
[~, cons_idx] = sort(consensus_imp, 'descend');
top_8 = cons_idx(1:min(8, length(cons_idx)));
barh(consensus_imp(top_8), 'FaceColor', natureColors.purple, ...
    'FaceAlpha', 0.8, 'EdgeColor', 'w');
set(gca, 'YTick', 1:length(top_8), 'YTickLabel', featureNames(top_8));
xlabel('Consensus Importance');
title('Top Features', 'FontWeight', 'bold');
grid on; box on;

subplot(2,2,4);
complexity_scores = [rf_nTrees * best_rf_mtry; 
                     best_lsb_learners * 2^best_lsb_depth;
                     1; 1];
scatter(complexity_scores, overall_scores_pct, 150, ...
    [modelColors.optrf; modelColors.lsb; modelColors.lasso; modelColors.ridge], ...
    'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
for ii = 1:4
    text(complexity_scores(ii)*1.3, overall_scores_pct(ii), model_names_display{ii}, ...
        'FontSize', 9, 'FontWeight', 'bold');
end
set(gca, 'XScale', 'log');
xlabel('Model Complexity'); ylabel('Performance Score');
title('Efficiency vs Performance', 'FontWeight', 'bold');
grid on; box on;

sgtitle('Figure 16: Performance Summary Dashboard', 'FontSize', 13, 'FontWeight', 'bold');

fprintf('\n========================================\n');
fprintf('ALL 16 FIGURES COMPLETE!\n');
fprintf('========================================\n');

[~, best_model_idx] = min(test_rmses);
fprintf('\nBest Model: %s\n', model_names_display{best_model_idx});
fprintf('  Test RMSE: %.4f Ah\n', min(test_rmses));
fprintf('  R2: %.4f\n', test_r2s(best_model_idx));

fprintf('\n========================================\n');
fprintf('CRITICAL NOTES FOR PAPER (V5 FIXES):\n');
fprintf('========================================\n');
fprintf('1. FIXED: All CV folds use independent normalization\n');
fprintf('2. FIXED: LOBO and final model use SAME parameter search space\n');
fprintf('3. FIXED: Learning curves use stratified early-cycle sampling\n');
fprintf('4. FIXED: Friedman test on battery-level LOBO results (N=4)\n');
fprintf('5. FIXED: Nemenyi test with correct q_alpha values\n');
fprintf('6. FIXED: Residual normality tests for all models\n');
fprintf('7. FIXED: VIF analysis with proper R2 truncation\n');
fprintf('8. FIXED: Unified permutation importance for all models\n');
fprintf('9. FIXED: Figure 8 labels TRAIN/TEST batteries\n');
fprintf('10. FIXED: Figure 10 renamed - NOT SHAP (Importance-Weighted Features)\n');
fprintf('11. FIXED: Figure 15 uses Model Disagreement Band (not CI)\n');
fprintf('12. FIXED: Q-Q plot with individual reference lines per model\n');
fprintf('13. FIXED: lightBlue color corrected to actual light blue\n');
fprintf('14. WARNING: Cycle feature importance should be discussed in paper\n');
fprintf('15. Limitations: Small dataset (N=4 batteries), single chemistry\n');
fprintf('========================================\n');

fprintf('\nLOBO Results for Paper Table:\n');
fprintf('%-10s | %-20s | %-20s\n', 'Model', 'RMSE (mean+/-std)', 'R2 (mean+/-std)');
fprintf('-----------|----------------------|---------------------\n');
for m = 1:4
    rmse_vals = lobo_results_all.(models_list{m}).rmse;
    r2_vals = lobo_results_all.(models_list{m}).r2;
    fprintf('%-10s | %.4f +/- %.4f      | %.4f +/- %.4f\n', ...
        model_names_display{m}, mean(rmse_vals), std(rmse_vals), ...
        mean(r2_vals), std(r2_vals));
end
fprintf('========================================\n');

function y = safe_normalize_vec(x)
    x = x(:);
    range_val = max(x) - min(x);
    if range_val < 1e-10
        y = 0.5 * ones(size(x));
    else
        y = (x - min(x)) / range_val;
    end
end