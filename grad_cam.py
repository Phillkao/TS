import seaborn as sns
from utils import *
from data import *


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activations = []

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, output):
        self.activations.append(output)

    def save_gradient(self, grad_output):
        # Gradients are computed in reverse order
        self.gradients = [grad_output[0]] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)


class BaseCAM:
    def __init__(self, model, target_layer, use_cuda=False):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layer)

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    @staticmethod
    def get_loss(output, target_category):
        return output[0][target_category]

    def __call__(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())
        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)
        # print(self.activations_and_grads.activations[-1]

        try:
            activations = self.activations_and_grads.activations[-1].cpu().data.numpy()[0, :]
        except:
            activations = self.activations_and_grads.activations[-1][0].cpu().data.numpy()[0, :]

        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()[0, :]
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :]
        cam = np.maximum(abs(cam), 0)
        heatmap = norm(cam)
        return heatmap


class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False):
        super(GradCAM, self).__init__(model, target_layer, use_cuda)

    def get_cam_weights(self, input_tensor,
                        target_category,
                        activations, grads):
        return np.mean(grads, axis=1)


def result_gradcam(test_loader, model, scaler, features_name, device='cuda', time_step=240, feature_size=49,
                   target_category=None, plot=True):
    results, target, _ = testing(model, test_loader, device=device)

    result_history = {'result': []}

    for idx, i in enumerate(test_loader):
        if results[idx] == target[idx]:
            target_layer_features = model.global_features_learning[-1]
            target_layer_tmp = model.temp_feature_learning[-1]
            target_layer_time = model.time_series_learning[-1]
            f_net = GradCAM(model, target_layer_features)
            t_net = GradCAM(model, target_layer_time)
            tm_net = GradCAM(model, target_layer_tmp)
            f_map = f_net(i[0].to(device), target_category=target_category - 1)
            t_map = t_net(i[0].to(device), target_category=target_category - 1)
            tm_map = tm_net(i[0].to(device), target_category=target_category - 1)
            org = scaler.transform(i[0].cpu().detach().numpy().reshape(time_step, feature_size))

            result_history['result'].append(
                (results[idx], f_map.T, tm_map.reshape(1, time_step), t_map.reshape(1, time_step)))

            if plot:
                print("------------------------%s/%s----------------------" % (str(results[idx]), str(target[idx])))
                fig, axs = plt.subplots(4, 1, figsize=(25, 25))
                sns.heatmap(tm_map.reshape(1, time_step), cmap=sns.cm.rocket_r, yticklabels=['Timestep'], ax=axs[0])
                axs[0].invert_xaxis()
                sns.heatmap(t_map.reshape(1, time_step), cmap=sns.cm.rocket_r, yticklabels=['Timestep'], ax=axs[1])
                axs[1].invert_xaxis()
                sns.heatmap(f_map.T, cmap=sns.cm.rocket_r, yticklabels=features_name, ax=axs[2])
                axs[2].invert_xaxis()
                sns.heatmap(org.T, cmap=sns.cm.rocket_r, yticklabels=features_name, ax=axs[3])
                axs[3].invert_xaxis()
                plt.tight_layout()
                plt.show()

    return result_history
