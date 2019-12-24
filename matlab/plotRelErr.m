% download the following data respository:
% https://physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip

% extract the .zip to directory structure /Data/EEG

clear;
sensInd = 1:64;
numInp = 32;

sInd = 1:109;
rInd = [3];
sInd = repmat(sInd, [length(rInd),1]);
sInd = reshape(sInd, 1, numel(sInd));
relErr = cell(length(sInd),2);
parfor i = 1:length(sInd)
    rIndInd = mod(i, length(rInd))+1;
    rIndUse = rInd(rIndInd);

    edfStr = sprintf('Data/EEG/S%03d/S%03dR%02d.edf',...
        sInd(i),sInd(i),rIndUse);
    patInd = sprintf('S%03dR%02d', sInd(i),rIndUse);
%     relErr{i,1} = patInd;
    fprintf('Model estimation for pat = %s\n', patInd);
    [~,~,~,~,errTemp] = modelEstNewv2('sensInd',sensInd,'numInp',numInp,'edfStr',edfStr,'silentFlag',1);
%     relErr{i,1} = patInd;
%     relErr{i,2} = errTemp;
    relErr(i,:) = {patInd, errTemp};
end
save('relErrFull_1step.mat', 'relErr');
a = relErr(:,2);
b = cell2mat(a);
b = b./repmat(sum(b,2),1,2);

% errorRatio = b(:,2)./b(:,1);
figure;
boxplot(b(:,2));
set(gca, 'ylim', [0,1]);
fprintf('Average error ratio = %f, error improvement=%f\n', mean(errorRatio), (1-mean(errorRatio))*100); 

figure;
h = bar(b(:,[2,1]), 'stacked','barWidth', 1);
h(1).FaceColor = [0 90 255]/255;
h(2).FaceColor = [255,74,63]/255;
h(1).EdgeColor = 'none';
h(2).EdgeColor = 'none';
set(gca, 'xlim', [0,length(sInd)]+0.5);
set(gca, 'ylim', [0,1]);

yCoordTemp = repmat(b(:,2),1,2);
yCoordTemp = reshape(yCoordTemp', 1, 2*size(yCoordTemp,1));
xCoordTemp = repmat([1:length(sInd)-1]',1,2);
xCoordTemp = reshape(xCoordTemp', 1, 2*size(xCoordTemp,1));
xCoordTemp = [0,xCoordTemp,length(sInd)]+0.5;

hold on;
plot(xCoordTemp, yCoordTemp, 'k', 'linewidth', 1);
xlabel('patient ID');
ylabel('relative error');

