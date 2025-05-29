import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication
import sys
import matplotlib.pyplot as plt

from scipy.stats import f
from sklearn.covariance import MinCovDet

class Test(QWidget):
    def __init__(self):
        super().__init__()
        self.df = None
    
    def load_csv(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Open CSV File", 
            "", 
            "CSV Files (*.csv)"
        )
        if not file_name:
            return

        self.df = pd.read_csv(file_name)
        if self.df.empty:
            print("Файл пуст или не содержит данных!")
            return

        # Попытка преобразовать строки в даты
        for col in self.df.columns:
            if pd.api.types.is_string_dtype(self.df[col]):
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                except:
                    pass

        # Переименование первого столбца (если он не "index")
        if not self.df.empty and self.df.columns[0] != "index":
            self.df.rename(columns={self.df.columns[0]: "index"}, inplace=True)

        # Округление чисел
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").round(4)

    # def find_stable_segment(self, data, min_length=30, var_threshold=0.3):
    #     """Ищет участок, где все параметры стабильны (по MAD), пропуская первые 2 столбца."""
    #     # Берем все столбцы, начиная с 3-го (индекс 2)
    #     value_columns = data.columns[2:]
        
    #     for i in range(len(data) - min_length):
    #         segment = data.iloc[i:i+min_length]
    #         mad = np.median(np.abs(segment[value_columns] - segment[value_columns].median()), axis=0)
    #         if (mad < var_threshold).mean() > 0.9 and np.median(mad) < var_threshold * 1.5:
    #             return segment
    #     raise ValueError("Стабильный участок не найден")
    def find_stable_segment(self, data, min_length=30, rel_threshold=0.05):  # rel_threshold = 5%
        value_columns = data.columns[2:]
        
        for i in range(len(data) - min_length):
            segment = data.iloc[i:i+min_length]
            median_vals = segment[value_columns].median()
            mad = np.median(np.abs(segment[value_columns] - median_vals), axis=0)
            rMAD = mad / np.abs(median_vals)  # Относительное отклонение
            
            if (rMAD < rel_threshold).mean() > 0.9 and np.median(rMAD) < rel_threshold * 1.5:
                return segment
        raise ValueError("Стабильный участок не найден")
    def adaptive_monitoring(self, min_length=20, var_threshold=0.1, 
                      l1=0.05, l2=0.01, transition_window=20, 
                      alpha=0.05):
        """
        Адаптивный мониторинг с подсчетом аномалий и переходных процессов
        """
        # Инициализация счетчиков
        total_anomalies = 0          # Общее количество аномалий
        total_transitions = 0        # Общее количество переходных процессов
        current_transition_length = 0 
        if self.df is None:
            raise ValueError("Сначала загрузите данные с помощью load_csv()")
        
        # Выбираем только числовые параметры (исключаем 'index' и 'time')
        param_columns = [col for col in self.df.columns 
                        if col not in ['index', 'time'] and 
                        pd.api.types.is_numeric_dtype(self.df[col])]
        
        if not param_columns:
            raise ValueError("Нет числовых параметров для анализа")
        
        data_values = self.df[param_columns].values  # Только числовые значения
        timestamps = self.df['time']  # Сохраняем временные метки
        
        p = len(param_columns)  # количество параметров
        
        # 1. Инициализация на стабильном сегменте
        stable_segment = self.find_stable_segment(self.df, min_length, var_threshold)
        if stable_segment is None or len(stable_segment) < min_length:
            raise ValueError(f"Не удалось найти стабильный сегмент длиной {min_length}")
        
        stable_values = stable_segment[param_columns].values
        
        # 2. Начальные параметры модели
        try:
            robust_cov = MinCovDet(support_fraction=0.75).fit(stable_values).covariance_
        except Exception as e:
            print(f"Ошибка MinCovDet: {e}, используем стандартную ковариацию")
            robust_cov = np.cov(stable_values, rowvar=False)
        
        mean = np.median(stable_values, axis=0)
        cov = robust_cov.copy()
        n_eff = max(int(2/l1), p+1)  # эффективный объем выборки
        
        transition_flag = False
        anomaly_counter = 0
        results = []
        
        # 3. Адаптивный мониторинг
        for t in range(len(stable_segment), len(data_values)):
            x_t = data_values[t]
            
            try:
                # 4. Расчет статистики T²
                diff = x_t - mean
                inv_cov = np.linalg.pinv(cov + 1e-6*np.eye(p))  # регуляризация
                T2 = diff @ inv_cov @ diff
                
                # 5. F-преобразование
                F = (n_eff - p) * T2 / (p * (n_eff - 1)) if n_eff > p else T2
                F_threshold = f.ppf(1-alpha, p, n_eff-p) * 2
                
                # 6. Детектирование аномалий
                is_anomaly = F > F_threshold
                results.append({
                    'timestamp': timestamps.iloc[t],
                    'F': F,
                    'threshold': F_threshold,
                    'anomaly': is_anomaly,
                    'transition': transition_flag,
                    'transition_id': total_transitions if transition_flag else 0,
                    'anomaly_count': total_anomalies,
                    **dict(zip(param_columns, x_t))  # сохраняем значения параметров
                })
                
                # 7. Логика переходных процессов
                if is_anomaly:
                    total_anomalies += 1
                    anomaly_counter += 1
                    if anomaly_counter >= transition_window and not transition_flag:
                        total_transitions += 1 
                        print(f"Обнаружен переходный процесс в {timestamps.iloc[t]}")
                        transition_flag = True
                        anomaly_counter = 0
                else:
                    anomaly_counter = max(0, anomaly_counter - 1)
                if transition_flag:
                    current_transition_length += 1
                # 8. Обновление параметров модели
                if not transition_flag:
                    # Экспоненциальное сглаживание
                    mean = l1 * x_t + (1 - l1) * mean
                    
                    if not is_anomaly:
                        residual = x_t - mean
                        cov = l2 * np.outer(residual, residual) + (1 - l2) * cov
                else:
                    # Проверка окончания переходного процесса
                    if t + transition_window <= len(data_values):
                        window_values = data_values[t:t+transition_window]
                        mad = np.median(np.abs(window_values - np.median(window_values, axis=0)), axis=0)
                        if (mad < var_threshold).all():
                            mean = np.median(window_values, axis=0)
                            cov = np.cov(window_values, rowvar=False)
                            transition_flag = False
                            print(f"Новый стабильный режим с {timestamps.iloc[t]}")
                            n_eff = max(int(2/l1), p+1)
            
            except Exception as e:
                print(f"Ошибка на шаге {t}: {str(e)}")
                continue
        stats = {
        'total_anomalies': total_anomalies,
        'total_transitions': total_transitions,
        'avg_transition_length': current_transition_length / max(1, total_transitions),
        'anomaly_rate': total_anomalies / len(data_values)
        }
        self.results = pd.DataFrame(results).set_index('timestamp')

        # Визуализация статистики
        self.plot_monitoring_stats(stats)
        print(total_transitions)
        
        return self.results, stats
    
    def plot_monitoring_stats(self, stats):
        """Визуализация статистики мониторинга"""
        plt.figure(figsize=(12, 6))
        
        # График 1: Общая статистика
        plt.subplot(1, 2, 1)
        plt.bar(['Аномалии', 'Переходы'], 
                [stats['total_anomalies'], stats['total_transitions']])
        plt.title('Общее количество событий')
        plt.ylabel('Количество')
        
        # График 2: Детализация
        plt.subplot(1, 2, 2)
        plt.pie([stats['anomaly_rate'], 1-stats['anomaly_rate']],
                labels=['Аномалии', 'Норма'],
                autopct='%1.1f%%')
        plt.title('Распределение аномалий')
        
        plt.tight_layout()
        plt.show()



    def plot_results(self):
        if not hasattr(self, 'results'):
            raise ValueError("Сначала выполните adaptive_monitoring()")
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.results.index, self.results['F'], label='F-статистика')
        plt.plot(self.results.index, self.results['threshold'], 'r--', label='Порог')
        plt.scatter(self.results[self.results['anomaly']].index,
                self.results[self.results['anomaly']]['F'],
                color='red', label='Аномалии')
        plt.legend()
        plt.title('Результаты мониторинга')
        plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    monitor = Test()
    monitor.load_csv()

    if monitor.df is not None:
        print("Загруженные данные:")
        print(monitor.df.head())
        
        stable_seg = monitor.find_stable_segment(monitor.df)
        print("\nСтабильный сегмент:")
        print(stable_seg)
        
        results = monitor.adaptive_monitoring()
        monitor.plot_results()
    
    sys.exit(app.exec_())
