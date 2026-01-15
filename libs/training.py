import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

def mutual_training_model_v3(root_dir, model1, model2, model3, train_loader, valid_loader, num_epochs=15, learning_rate=0.001, device='cuda'):
    model1 = model1.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model1_optimizer = optim.AdamW(model1.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.999))
    model2_optimizer = optim.AdamW(model2.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.999))
    model3_optimizer = optim.AdamW(model3.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.999))

    scheduler_model1 = optim.lr_scheduler.OneCycleLR(
        model1_optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25,
        final_div_factor=250,
        anneal_strategy='cos'
    )

    scheduler_model2 = optim.lr_scheduler.OneCycleLR(
        model2_optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25,
        final_div_factor=250,
        anneal_strategy='cos'
    )

    scheduler_model3 = optim.lr_scheduler.OneCycleLR(
        model3_optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25,
        final_div_factor=250,
        anneal_strategy='cos'
    )

    best_val_acc = 0.0
    patience = 3
    counter = 0

    scaler = torch.amp.GradScaler(device=device,
                                  init_scale=2**16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        model1.train()
        model2.train()
        model3.train()

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            model1_optimizer.zero_grad()
            model2_optimizer.zero_grad()
            model3_optimizer.zero_grad()

            with torch.amp.autocast(device):
                model1_outputs = model1(inputs)
                model2_outputs = model2(inputs)
                model3_outputs = model3(inputs)

                soft_labels_from_model2 = torch.softmax(model2_outputs.detach(), dim=1)
                soft_labels_from_model1 = torch.softmax(model1_outputs.detach(), dim=1)
                soft_labels_from_model3 = torch.softmax(model3_outputs.detach(), dim=1)

                model1_loss = criterion(model1_outputs, labels) + 0.5 * torch.mean(
                      torch.sum(soft_labels_from_model2 * torch.log(soft_labels_from_model2 / soft_labels_from_model1 + 1e-6), dim=1)
                ) + 0.5 * torch.mean(
                      torch.sum(soft_labels_from_model3 * torch.log(soft_labels_from_model3 / soft_labels_from_model1 + 1e-6), dim=1)
                )

                model2_loss = criterion(model2_outputs, labels) + 0.5 * torch.mean(
                      torch.sum(soft_labels_from_model1 * torch.log(soft_labels_from_model1 / soft_labels_from_model2 + 1e-6), dim=1)
                ) + 0.5 * torch.mean(
                      torch.sum(soft_labels_from_model3 * torch.log(soft_labels_from_model3 / soft_labels_from_model2 + 1e-6), dim=1)
                )

                model3_loss = criterion(model3_outputs, labels) + 0.5 * torch.mean(
                      torch.sum(soft_labels_from_model1 * torch.log(soft_labels_from_model1 / soft_labels_from_model3 + 1e-6), dim=1)
                ) + 0.5 * torch.mean(
                      torch.sum(soft_labels_from_model2 * torch.log(soft_labels_from_model2 / soft_labels_from_model3 + 1e-6), dim=1)
                )

            scaler.scale(model1_loss).backward()
            scaler.scale(model2_loss).backward()
            scaler.scale(model3_loss).backward()

            torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model3.parameters(), max_norm=1.0)

            scaler.step(model1_optimizer)
            scaler.step(model2_optimizer)
            scaler.step(model3_optimizer)
            scaler.update()
            
            scheduler_model1.step()
            scheduler_model2.step()
            scheduler_model3.step()

            _, preds1 = torch.max(model1_outputs, 1)
            _, preds2 = torch.max(model2_outputs, 1)
            _, preds3 = torch.max(model3_outputs, 1)
            running_loss += ((model1_loss.item() + model2_loss.item() + model3_loss.item())/3) * inputs.size(0)
            running_corrects += (torch.sum(preds1 == labels.data).item() + torch.sum(preds2 == labels.data).item() + torch.sum(preds3 == labels.data).item()) / 3
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        model1.eval()
        model2.eval()
        model3.eval()
        val_running_loss = 0.0
        val_running_corrects1, val_running_corrects2, val_running_corrects3 = 0, 0, 0
        val_total_samples = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                model1_outputs = model1(inputs)
                model2_outputs = model2(inputs)
                model3_outputs = model3(inputs)

                model1_preds = torch.argmax(model1_outputs, dim=1)
                model2_preds = torch.argmax(model2_outputs, dim=1)
                model3_preds = torch.argmax(model3_outputs, dim=1)

                loss1 = criterion(model1_outputs, labels)
                loss2 = criterion(model2_outputs, labels)
                loss3 = criterion(model3_outputs, labels)
                loss = (loss1 + loss2 + loss3) / 3

                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects1 += torch.sum(model1_preds == labels.data).item()
                val_running_corrects2 += torch.sum(model2_preds == labels.data).item()
                val_running_corrects3 += torch.sum(model3_preds == labels.data).item()
                val_total_samples += inputs.size(0)

        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_acc = (val_running_corrects1 + val_running_corrects2 + val_running_corrects3)  / (3*val_total_samples)
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc*100:.2f}% - Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_acc*100:.2f}%")

        # scheduler_model1.step()
        # scheduler_model2.step()
        # scheduler_model3.step()

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model1.state_dict(), os.path.join(root_dir, 'libs/best_model1.pth'))
            torch.save(model2.state_dict(), os.path.join(root_dir, 'libs/best_model2.pth'))
            torch.save(model3.state_dict(), os.path.join(root_dir, 'libs/best_model3.pth'))
            print(f"Saved best models with validation accuracy: {best_val_acc*100:.4f}")
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss vs Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy vs Epochs')
    plt.savefig(os.path.join(root_dir, 'libs/training_history.png'))
    plt.show()

    return model1, model2, model3