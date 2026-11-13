using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

Console.OutputEncoding = Encoding.UTF8;

var ml = new MLContext(seed: 42);
var dataPath = Path.Combine(AppContext.BaseDirectory, "data.csv");
var data = ml.Data.LoadFromTextFile<Borrower>(
    dataPath, separatorChar: ',', hasHeader: true);

// Пропущенный стаж заменяем средним стажем обучающей выборки.
// Вектор содержит только признаки клиента. Нормализация приводит
// доход, возраст и доли к сопоставимому масштабу перед обучением.
var pipeline = ml.Transforms.ReplaceMissingValues(
        "EmploymentYearsFilled", nameof(Borrower.EmploymentYears),
        MissingValueReplacingEstimator.ReplacementMode.Mean)
    .Append(ml.Transforms.Concatenate("Features",
        nameof(Borrower.Age), nameof(Borrower.Income),
        "EmploymentYearsFilled", nameof(Borrower.CreditHistory),
        nameof(Borrower.DebtToIncomeRatio)))
    .Append(ml.Transforms.NormalizeMinMax("Features"))
    .Append(ml.BinaryClassification.Trainers.SdcaLogisticRegression(
        labelColumnName: nameof(Borrower.Target)));

Console.WriteLine("Лабораторная работа № 10, вариант 1");
Console.WriteLine("Оценка риска кредитного заёмщика");
Console.WriteLine();

// Для оценки обучаем отдельную модель. Тестовые строки не участвуют
// в заполнении пропусков, нормализации и обучении классификатора.
var split = ml.Data.TrainTestSplit(data, testFraction: 0.2, seed: 42);
var validationModel = pipeline.Fit(split.TrainSet);
var testPredictions = validationModel.Transform(split.TestSet);
var metrics = ml.BinaryClassification.Evaluate(
    testPredictions, labelColumnName: nameof(Borrower.Target));

var trainCount = ml.Data.CreateEnumerable<Borrower>(split.TrainSet, false).Count();
var testCount = ml.Data.CreateEnumerable<Borrower>(split.TestSet, false).Count();
Console.WriteLine($"Записей в CSV: {trainCount + testCount}");
Console.WriteLine($"Проверка качества: {trainCount} обучающих и {testCount} тестовых записей");
Console.WriteLine($"Accuracy: {metrics.Accuracy:P1}");
Console.WriteLine();

// Для итоговых прогнозов заново обучаем модель на всех строках CSV.
var model = pipeline.Fit(data);
using var predictor = ml.Model.CreatePredictionEngine<Borrower, RepaymentPrediction>(model);
Borrower[] newBorrowers =
[
    new() { Age = 34, Income = 4600, EmploymentYears = 9, CreditHistory = 0, DebtToIncomeRatio = 0.09f },
    new() { Age = 28, Income = 1750, EmploymentYears = 2, CreditHistory = 1, DebtToIncomeRatio = 0.43f },
    new() { Age = 37, Income = 3250, EmploymentYears = 7, CreditHistory = 0, DebtToIncomeRatio = 0.19f }
];

Console.WriteLine("Прогнозы для трёх новых клиентов");
for (int index = 0; index < newBorrowers.Length; index++)
{
    var borrower = newBorrowers[index];
    var prediction = predictor.Predict(borrower);
    string label = prediction.WillRepay ? "да" : "нет";

    Console.WriteLine($"Клиент {index + 1}: возраст {borrower.Age}, доход {borrower.Income} у.е., " +
        $"стаж {borrower.EmploymentYears} лет, прошлый дефолт {borrower.CreditHistory}, " +
        $"кредит/годовой доход {borrower.DebtToIncomeRatio:P0}");
    Console.WriteLine($"  Вернёт кредит: {label}. Вероятность возврата: {prediction.Probability:P2}");
}

// LoadColumn связывает свойства с индексами столбцов в CSV.
// ML.NET читает числовую метку 0/1 в логическое свойство Target.
public sealed class Borrower
{
    [LoadColumn(0)] public float Age { get; set; }
    [LoadColumn(1)] public float Income { get; set; }
    [LoadColumn(2)] public float EmploymentYears { get; set; }
    [LoadColumn(3)] public float CreditHistory { get; set; }
    [LoadColumn(4)] public float DebtToIncomeRatio { get; set; }
    [LoadColumn(5)] public bool Target { get; set; }
}

public sealed class RepaymentPrediction
{
    [ColumnName("PredictedLabel")] public bool WillRepay { get; set; }
    public float Probability { get; set; }
}
