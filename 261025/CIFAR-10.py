# TODO: CUDA GPU 장치에서 CIFAR-10 데이터셋을 학습과 평가
# TODO: 각 Epoch 및 전체 학습 시간을 출력

EPOCHS = 5

# 학습 기록 저장 공간 준비
train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []

# 전체 학습 시간 측정 시작
torch.cuda.synchronize()
training_start_time = time.perf_counter()

# TODO: 5번의 Epoch 동안 학습 및 평가 진행
for epoch in range(EPOCHS):
    # 에포크 시작 시간 측정 (GPU 동기화 포함)
    torch.cuda.synchronize()
    epoch_start_time = time.perf_counter()

    # 1. 1개 에포크 학습
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    # 2. 모델 평가
    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    # 3. 기록 저장
    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)
    test_loss_history.append(test_loss)
    test_accuracy_history.append(test_accuracy)

    # 에포크 종료 시간 측정
    torch.cuda.synchronize()
    epoch_time = time.perf_counter() - epoch_start_time

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: "
        f"{train_accuracy * 100:.2f}% | "
        f"Test loss: {test_loss:.4f} | "
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}% | "
        f"Time: {epoch_time:.2f}s"
    )

# TODO: 전체 학습 시간 계산 및 출력
torch.cuda.synchronize()
total_training_time = time.perf_counter() - training_start_time

print(f"Total GPU training time: {total_training_time:.2f}s")