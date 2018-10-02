mat1 = struct2cell(load('banknote-traindata'));
training = mat1{1};

count_y = 0;
count_y0 = 0;

for i=1:size(training, 1)
    if(training(i,5) == 1)
        count_y = count_y + 1;
    end
    if(training(i,5) == 0)
        count_y0 = count_y0 + 1;
    end        
end
pi_0 = count_y0/(count_y0+count_y);
pi_1 = count_y/(count_y0+count_y);


%evaluate mean and std for all the variables for the two classes 
%{yes,no} == {1,0}. yes -> genuine notes. no ->  fake notes

% For simplicity the training array has been filtered for yes's and no's
% and now the mean and std can easily be evaluated for the two classes
training1 = ones(487, 5);
training0 = ones(611, 5);

count1 = 1; 
count0 = 1;
for k=1:1098
    if training(k, 5) == 1
        training1(count1,:) = training(k,:);
        count1 = count1 + 1;
    else
        training0(count0,:) = training(k,:);
        count0 = count0 + 1;
    end
end

count1 = count1 - 1;
count0 = count0 - 1;
% Decrement the variables to keep track.
meanYes = zeros(1,4);
meanNo = zeros(1,4);

for i=1:4
    meanYes(1,i) = mean(training1(:,i));
    meanNo(1,i) = mean(training0(:,i));
end

% For each class we will compute the covariance matrix
% The covariance will be a 4 by 4 matrix since we have 4 features involved
% here. Sigma is the covariance matrix. 
sigma_yes = cov(training1(:,1:4));
sigma_no = cov(training0(:,1:4));
 
% We can say at this point that the training set has been learnt
% in the form of the parameters of the multivariate gaussian distribution
% lets now classify the dataset using the decision boundary function
% discussed in class and label the test data set.

mat2 = struct2cell(load('banknote-testdata'));
test = mat2{1};

% decision = -2*log(pi_1/pi_0) - 0.5*(transpose(meanYes))*(inv(sigma_yes))*(meanYes)+0.5*(transpose(meanNo))*inv(sigma_no)*meanNo + (transpose(meanYes-meanNo))*inv(sigmaYes)  
results = ones(274,1);
meanYes = meanYes';
meanNo = meanNo';
probMat = ones(274,1);
for i=1:274 %test sample size
    x = transpose(test(i,1:4));
    s2 = inv(sigma_yes);
    s1 = inv(sigma_no);
    m2 = meanYes';
    m1 = meanNo';
    term = 2*(transpose(s2*meanYes - s1*meanNo))*x;
    detVal = det(sigma_yes)/det(sigma_no);
    X = transpose(x);
    decision = X*(s1-s2)+term+(m1*s1*meanNo)-(m2*s2*meanYes)-log(detVal)-2*log(pi_0/pi_1);
    
%     for the confusion matrix
    term1 = (1/power((2*pi),4/2))*(1/det(sigma_yes));
    a = transpose(x-meanYes);
    term2 = exp(-0.5*a*s2*(x-meanYes));
    probMat(i,1) = term1*term2;
    if decision > 1
        results(i) = 1;
    else
        results(i) = 0;
    end
end

output = ones(274,2);
output(:,1) = test(:,5);
output(:,2) = results;
% For the confusion matrix we will have to find the probabilities of
% getting a yes using the equation of a mutltivariate normal distribution.
% this has been done in the loop above but we have to normalize probMat
% before we use the plotconfusion function
for l=1:274
    probMat(l,1) = probMat(l,1)/sum(probMat(:,1));
end

probMat = probMat';

[A,B ,C,D] = confusion(transpose(test(:, 5)),probMat(1,:));
plotconfusion(transpose(test(:, 5)),probMat(1,:));
% pause;
plotroc(transpose(test(:, 5)),probMat(1,:));
