% ============================================================
% Script: find_archetypes_PTI_dkfz.m
%
% Purpose:
%   - Remove three outlier samples from DKFZ Wegert dataset
%   - (Optionally) write an outlier-free expression matrix to CSV
%   - Run Pareto Task Inference (ParTI) to find archetypes
%   - Rename archetype variables to biologically meaningful labels
%
% Inputs (relative paths):
%   ../data/GSE53224_series_matrix.xlsx
%   ../data/ordered_metadata_gessler.xlsx
%   ../data/norm_data_dfkz_2019-10-30.csv
%
% Outputs (optional):
%   ../data/gene_expression_dkfz_noOutliers.csv
%   ../user_output/geneExpression_dkfz_arcs.csv (if enabled)
%
% Dependencies:
%   - ParTI code (Uri Alon lab) in ../ParTI_uri_alon_lab/...
%   - Custom function find_Arcs.m (expects a CSV filepath)
% ============================================================

clear; clc;

%% ------------------------------------------------------------------------
%  Parameters
% -------------------------------------------------------------------------
WRITE_EXPR_NO_OUTLIERS = false;  % write expression matrix without outliers
WRITE_FINAL_TABLE      = false;  % write final table with archetypes

expr_input_file   = '../data/norm_data_dfkz_2019-10-30.csv';
expr_output_file  = '../data/gene_expression_dkfz_noOutliers.csv';  % used by ParTI
final_output_file = '../user_output/geneExpression_dkfz_arcs.csv';

meta_file_1 = '../data/GSE53224_series_matrix.xlsx';
meta_file_2 = '../data/ordered_metadata_gessler.xlsx';

outliers = {'GSM1287965', 'GSM1287967', 'GSM1287968'};  % GSM IDs to remove

%% ------------------------------------------------------------------------
%  Load and clean metadata (remove outliers)
% -------------------------------------------------------------------------
% metadata: first 30 rows, skip first column (assumed non-sample column)
metadata   = readtable(meta_file_1);
metadata   = metadata(1:30, 2:end);

metadata_2 = readtable(meta_file_2);

% Find columns corresponding to outlier GSM IDs (assumes they appear in row 1)
[~, idx_out_meta] = ismember(outliers, metadata{1, :});
idx_out_meta(idx_out_meta == 0) = [];  % remove non-matches, just in case

% Remove outliers from metadata
metadata(:, idx_out_meta) = [];

% Remove same samples from metadata_2 (assumes matching order)
metadata_2(idx_out_meta, :) = [];

%% ------------------------------------------------------------------------
%  Load expression data and remove outlier samples
% -------------------------------------------------------------------------
geneExpression_dkfz = readtable(expr_input_file, 'ReadRowNames', true);

% Remove outlier columns by variable name
geneExpression_dkfz.GSM1287965_dkfz1247_CEL_gz = [];
geneExpression_dkfz.GSM1287967_dkfz1250_CEL_gz = [];
geneExpression_dkfz.GSM1287968_dkfz1251_CEL_gz = [];

% Optionally write expression matrix without outliers
if WRITE_EXPR_NO_OUTLIERS
    writetable(geneExpression_dkfz, expr_output_file, 'WriteRowNames', true);
end

%% ------------------------------------------------------------------------
%  Add ParTI code to path
% -------------------------------------------------------------------------
addpath( ...
    '../ParTI_uri_alon_lab', ...
    '../ParTI_uri_alon_lab/PCHA', ...
    '../ParTI_uri_alon_lab/ADVMM_and_SDVMM_codes', ...
    '../ParTI_uri_alon_lab/sisal_demo', ...
    '../ParTI_uri_alon_lab/SeDuMi_1_3' ...
);

%% ------------------------------------------------------------------------
%  Run Pareto Task Inference (ParTI)
%  Note: find_Arcs should internally call ParTI on the given CSV.
%        Ensure expr_output_file exists (either pre-generated or by
%        setting WRITE_EXPR_NO_OUTLIERS = true and re-running).
% -------------------------------------------------------------------------
filePath = expr_output_file;  % the file ParTI will use

[arcs, data] = find_Arcs(filePath); %#ok<ASGLU>
% 'data' is assumed to be a table with archetype coordinates appended
% as the last columns.

%% ------------------------------------------------------------------------
%  Rename archetype variables to meaningful biological labels
% -------------------------------------------------------------------------
n_arcs = 3;  % number of archetypes in the dataset

% Current variable names in 'data'
varNames = data.Properties.VariableNames;

% Replace last n_arcs variable names with archetype labels
newArcNames = {'Arc1_Stromal', 'Arc2_Epi', 'Arc3_Blast'};
if numel(varNames) < n_arcs
    error('Not enough columns in ''data'' to rename archetypes.');
end

varNames(end - n_arcs + 1:end) = newArcNames;
data.Properties.VariableNames = varNames;

%% ------------------------------------------------------------------------
%  Optionally write final table with archetype labels
% -------------------------------------------------------------------------
if WRITE_FINAL_TABLE
    writetable(data, final_output_file, 'WriteRowNames', true);
end
