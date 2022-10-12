import torch
import torch.nn as nn
import torch.nn.functional as F


class same_padding(nn.Module):
    def __init__(self, kernel_size) -> None:
        super(same_padding, self).__init__()
        self.pad_size = int(kernel_size / 2 - 1)

    def forward(self, inputs):
        return F.pad(inputs, (0, 0, self.pad_size, self.pad_size))


class Upsampling1D(torch.nn.Module):
    def __init__(self, scale_factor):
        super(Upsampling1D, self).__init__()
        self.upsampling2D = nn.UpsamplingNearest2d(scale_factor=scale_factor)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.upsampling2D(x)

        return x


class MTEX_CNN(nn.Module):
    def __init__(self, input_shape, n_class):
        super(MTEX_CNN, self).__init__()

        self.time_step = input_shape[0]
        self.features_size = input_shape[1]

        self.global_features_learning = nn.Sequential(

            same_padding(kernel_size=8),
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(8, 1), stride=(2, 1)),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            same_padding(kernel_size=6),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(2, 1)),
            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.ReLU(),
        )

        self.time_series_learning = nn.Sequential(
            nn.Conv1d(in_channels=self.features_size, out_channels=128,
                      kernel_size=4, stride=2, padding='valid'),
            nn.ReLU(),
            nn.Dropout(p=0.4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(14 * 128, 128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.global_features_learning(x)
        x = torch.reshape(x, (-1, self.features_size, int(self.time_step / 4)))
        x = self.time_series_learning(x)
        x = torch.reshape(x, (-1, 14 * 128))
        x = self.classifier(x)

        return x


class XCM(nn.Module):
    def __init__(self, input_shape, n_class, window_size=0.2, filters_num=128):
        super(XCM, self).__init__()

        self.time_step = input_shape[0]
        self.features_size = input_shape[1]
        self.filters_num = filters_num

        self.global_features_learning = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=self.filters_num,
                      kernel_size=(int(window_size * self.time_step), 1), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(self.filters_num),
            nn.ReLU(),

            # nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (6,1), stride = (2,1)),
            # nn.ReLU(),
            # nn.Dropout(p=0.4),

            nn.Conv2d(in_channels=self.filters_num, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.ReLU(),
        )

        self.temp_feature_learning = nn.Sequential(
            nn.Conv1d(in_channels=self.features_size, out_channels=self.filters_num,
                      kernel_size=int(window_size * self.time_step), stride=1, padding='same'),
            nn.BatchNorm1d(self.filters_num),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filters_num, out_channels=1, kernel_size=1,
                      stride=1),
            nn.ReLU(),
        )

        self.time_series_learning = nn.Sequential(
            nn.Conv1d(in_channels=self.features_size + 1, out_channels=self.filters_num,
                      kernel_size=int(window_size * self.time_step), stride=1, padding='same'),
            nn.BatchNorm1d(self.filters_num),
            nn.ReLU(),
        )

        self.global_avg_pool = nn.AvgPool2d((1, self.time_step))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.filters_num, out_features=n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        temp_x = torch.reshape(x, (-1, self.features_size, self.time_step))
        x = self.global_features_learning(x)
        x = torch.reshape(x, (-1, self.features_size, self.time_step))
        temp_x = self.temp_feature_learning(temp_x)
        z = torch.concat((x, temp_x), dim=1)

        z = self.time_series_learning(z)
        z = self.global_avg_pool(z)
        z = torch.reshape(z, (-1, self.filters_num))
        z = self.classifier(z)

        return z


class TSEM(nn.Module):
    def __init__(self, input_shape, n_class, window_size=0.2, filters_num=128):
        super(TSEM, self).__init__()

        self.time_step = input_shape[0]
        self.features_size = input_shape[1]
        self.filters_num = filters_num
        self.hidden = int(window_size * self.time_step)

        self.global_features_learning = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=self.filters_num,
                      kernel_size=(int(window_size * self.time_step), 1), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(self.filters_num),
            nn.ReLU(),

            nn.Conv2d(in_channels=self.filters_num, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.ReLU(),
        )

        self.temp_feature_learning = nn.LSTM(input_size=self.features_size,
                                             hidden_size=int(window_size * self.time_step), num_layers=1,
                                             batch_first=True)

        self.upsampling = Upsampling1D(scale_factor=5)

        self.time_series_learning = nn.Sequential(
            nn.Conv1d(in_channels=self.features_size, out_channels=self.filters_num,
                      kernel_size=int(window_size * self.time_step), stride=1, padding='same'),
            nn.BatchNorm1d(self.filters_num),
            nn.ReLU(),
        )

        self.global_avg_pool = nn.AvgPool2d((1, self.time_step))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.filters_num, out_features=n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        temp_x = torch.reshape(x, (-1, self.time_step, self.features_size))
        x = self.global_features_learning(x)
        x = torch.reshape(x, (-1, self.features_size, self.time_step))
        temp_x, (h_out, c_out) = self.temp_feature_learning(temp_x)
        temp_x = h_out[:, -1, :]
        temp_x = self.upsampling(temp_x)

        z = x * temp_x
        z = self.time_series_learning(z)
        z = self.global_avg_pool(z)
        z = torch.reshape(z, (-1, self.filters_num))
        z = self.classifier(z)

        return z


class XCM_seq(nn.Module):
    def __init__(self, input_shape, n_class, window_size=0.2, filters_num=128):
        super(XCM_seq, self).__init__()

        self.time_step = input_shape[0]
        self.features_size = input_shape[1]
        self.filters_num = filters_num

        self.global_features_learning = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=self.filters_num,
                      kernel_size=(int(window_size * self.time_step), 1), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(self.filters_num),
            nn.ReLU(),

            # nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (6,1), stride = (2,1)),
            # nn.ReLU(),
            # nn.Dropout(p=0.4),

            nn.Conv2d(in_channels=self.filters_num, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.ReLU(),
        )

        self.temp_feature_learning = nn.Sequential(
            nn.Conv1d(in_channels=self.features_size, out_channels=self.filters_num,
                      kernel_size=int(window_size * self.time_step), stride=1, padding='same'),
            nn.BatchNorm1d(self.filters_num),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.filters_num, out_channels=1, kernel_size=1,
                      stride=1),
            nn.ReLU(),
        )

        self.time_series_learning = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.filters_num,
                      kernel_size=int(window_size * self.time_step), stride=1, padding='same'),
            nn.BatchNorm1d(self.filters_num),
            nn.ReLU(),
        )

        self.global_avg_pool = nn.AvgPool2d((1, self.time_step))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.filters_num, out_features=n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # temp_x = torch.reshape(x, (-1, self.features_size, self.time_step))
        x = self.global_features_learning(x)
        x = torch.reshape(x, (-1, self.features_size, self.time_step))
        temp_x = torch.reshape(x, (-1, self.features_size, self.time_step))
        temp_x = self.temp_feature_learning(temp_x)
        # z = torch.concat((x, temp_x), dim = 1)
        z = temp_x

        z = self.time_series_learning(z)
        z = self.global_avg_pool(z)
        z = torch.reshape(z, (-1, self.filters_num))
        z = self.classifier(z)

        return z


class TSEM_seq(nn.Module):
    def __init__(self, input_shape, n_class, window_size=0.2, filters_num=128):
        super(TSEM_seq, self).__init__()

        self.time_step = input_shape[0]
        self.features_size = input_shape[1]
        self.filters_num = filters_num
        self.hidden = int(window_size * self.time_step)

        self.global_features_learning = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=self.filters_num,
                      kernel_size=(int(window_size * self.time_step), 1), stride=(1, 1), padding='same'),
            nn.BatchNorm2d(self.filters_num),
            nn.ReLU(),

            nn.Conv2d(in_channels=self.filters_num, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.ReLU(),
        )

        self.temp_feature_learning = nn.LSTM(input_size=self.features_size, hidden_size=self.time_step, num_layers=1,
                                             batch_first=True)

        # self.upsampling = Upsampling1D(scale_factor = 5)
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=5)
        self.time_series_learning = nn.Sequential(
            nn.Conv1d(in_channels=self.time_step, out_channels=self.filters_num,
                      kernel_size=int(window_size * self.time_step), stride=1, padding='same'),
            nn.BatchNorm1d(self.filters_num),
            nn.ReLU(),
        )

        self.global_avg_pool = nn.AvgPool2d((1, self.time_step))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.filters_num, out_features=n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.global_features_learning(x)
        x = torch.reshape(x, (-1, self.features_size, self.time_step))
        temp_x = torch.reshape(x, (-1, self.time_step, self.features_size))
        temp_x, (_, _) = self.temp_feature_learning(temp_x)
        # print(temp_x.size())
        temp_x = torch.reshape(temp_x, (-1, self.time_step, self.time_step))
        z = temp_x
        z = self.time_series_learning(z)
        z = self.global_avg_pool(z)
        z = torch.reshape(z, (-1, self.filters_num))
        z = self.classifier(z)

        return z
