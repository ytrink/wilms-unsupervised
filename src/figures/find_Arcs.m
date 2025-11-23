function [arcs, geneExpressionArcs] = find_Arcs(filename)
% find_Arcs  Run ParTI_lite on an expression matrix and append archetype columns
%
% INPUT
%   filename  path to a CSV with genes as row names and samples as columns
%
% OUTPUT
%   arcs                (k x G) numeric matrix of archetype gene profiles
%   geneExpressionArcs  table with original expression plus k archetype columns
%
    arguments
        filename (1, :) char
    end

    % Read expression (genes in rows, samples in columns)
    expressionTable = readtable(filename, 'ReadRowNames', true);
    if isempty(expressionTable)
        error('Input table is empty or unreadable: %s', filename);
    end

    % Convert to samples x genes for ParTI_lite
    expressionArray = table2array(expressionTable)';  % (S x G)

    % Run ParTI (assumes it chooses k, commonly 3)
    % If your version takes k, use: [arc, arcOrig, pc] = ParTI_lite(expressionArray, k);
    [arc, arcOrig, pc] = ParTI_lite(expressionArray); %#ok<ASGLU>

    % Prepare archetype columns as a table with gene row names
    % arcOrig is (k x G) so transpose to (G x k) to match expressionTable's (G x S)
    k = size(arcOrig, 1);
    arcNames = arrayfun(@(i) sprintf('Arc%d', i), 1:k, 'UniformOutput', false);

    arcTable = array2table(arcOrig', ...
        'RowNames', expressionTable.Properties.RowNames, ...
        'VariableNames', arcNames);

    % Concatenate by columns: original expression + archetype profiles
    geneExpressionArcs = [expressionTable arcTable];

    % Return archetypes as numeric matrix (k x G)
    arcs = arcOrig;
end
