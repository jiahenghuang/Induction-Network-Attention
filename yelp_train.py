import torch
from torch.autograd import Variable
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(attention_model, train_loader, criterion, optimizer, epochs=5, use_regularization=False, C=0, clip=False):
    """
        Training code
        Args:
            attention_model : {object} model
            train_loader    : {DataLoader} training data loaded into a dataloader
            optimizer       :  optimizer
            criterion       :  loss function. Must be BCELoss for binary_classification and NLLLoss for multiclass
            epochs          : {int} number of epochs
            use_regularizer : {bool} use penalization or not
            C               : {int} penalization coeff
            clip            : {bool} use gradient clipping or not
        Returns:
            accuracy and losses of the model
        """
    losses = []
    accuracy = []
    avg_sentence_embeddings=None

    for i in range(epochs):
        total_loss = 0
        n_batches = 0
        correct = 0

        for batch_idx, train in enumerate(train_loader):
            if batch_idx%10==0:
                print("epochs",epochs,"batch_idx",batch_idx)
            if train[0].shape[0]!=train_loader.batch_size:
                print("  train[0].shape[0]!=train_loader.batch_size! ")
                continue
            attention_model.hidden_state = attention_model.init_hidden()
            x, y = Variable(train[0]).to(device), Variable(train[1]).to(device)
            y_pred, att,avg_sentence_embeddings = attention_model(x)

            # penalization AAT - I
            if use_regularization:
                attT = att.transpose(1, 2)
                identity = torch.eye(att.size(1)).to(device)
                identity = Variable(identity.unsqueeze(0).expand(train_loader.batch_size, att.size(1), att.size(1)))
                penal = attention_model.l2_matrix_norm(att @ attT - identity)

            if not bool(attention_model.type):
                # binary classification
                # Adding a very small value to prevent BCELoss from outputting NaN's
                # correct += torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)), y).data.sum()
                correct += torch.eq(torch.round(y_pred.squeeze(1)), y).data.sum()
                if use_regularization:
                    try:
                        # loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1) + 1e-8,y) + C * penal / train_loader.batch_size
                        loss = criterion(y_pred.squeeze(1) + 1e-8,y) + C * penal / train_loader.batch_size

                    except RuntimeError:
                        raise Exception(
                            "BCELoss gets nan values on regularization. Either remove regularization or add very small values")
                else:
                    # loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1), y)
                    loss = criterion(y_pred.squeeze(1), y)


            else:
                # correct += torch.eq(torch.max(y_pred, 1)[1], y.type(torch.LongTensor)).data.sum()
                correct += torch.eq(torch.max(y_pred, 1)[1], y).data.sum()
                if use_regularization:
                    # loss = criterion(y_pred, y) + (C * penal / train_loader.batch_size).type(torch.FloatTensor)
                    loss = criterion(y_pred, y) + (C * penal / train_loader.batch_size)
                else:
                    loss = criterion(y_pred, y)

            total_loss += loss.data
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping
            if clip:
                torch.nn.utils.clip_grad_norm(attention_model.parameters(), 0.5)
            optimizer.step()
            n_batches += 1

        print("avg_loss is", total_loss / n_batches)
        print("Accuracy of the model", correct / (n_batches * train_loader.batch_size))
        losses.append(total_loss / n_batches)
        accuracy.append(correct / (n_batches * train_loader.batch_size))
    return losses, accuracy,avg_sentence_embeddings


def evaluate(attention_model, x_test, y_test):
    """
        cv results
        Args:
            attention_model : {object} model
            x_test          : {nplist} x_test
            y_test          : {nplist} y_test

        Returns:
            cv-accuracy
    """

    attention_model.batch_size = x_test.shape[0]
    attention_model.hidden_state = attention_model.init_hidden()
    # x_test_var = Variable(torch.from_numpy(x_test).type(torch.LongTensor))
    x_test_var = x_test
    y_test_pred, _ = attention_model(x_test_var)
    if bool(attention_model.type):
        y_preds = torch.max(y_test_pred, 1)[1]
        # y_test_var = Variable(torch.from_numpy(y_test).type(torch.LongTensor))
        y_test_var = y_test

    else:
        # y_preds = torch.round(y_test_pred.type(torch.DoubleTensor).squeeze(1))
        # y_test_var = Variable(torch.from_numpy(y_test).type(torch.DoubleTensor))
        y_preds = torch.round(y_test_pred.squeeze(1))
        y_test_var = y_test

    return torch.eq(y_preds, y_test_var).data.sum() / x_test_var.size(0)


def get_activation_wts(attention_model, x):
    """
        Get r attention heads
        Args:
            attention_model : {object} model
            x               : {torch.Variable} input whose weights we want
        Returns:
            r different attention weights
    """
    attention_model.batch_size = x.size(0)
    attention_model.hidden_state = attention_model.init_hidden()
    _, wts = attention_model(x)


    return wts