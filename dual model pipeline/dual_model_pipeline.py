import cv2
import torch
import numpy
import matplotlib.pyplot
from pathlib import Path
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import tkinter
from tkinter import ttk, filedialog
import logic

class Visualiser:
    def __init__(self):
        matplotlib.pyplot.rcParams.update({"figure.facecolor": "white", "axes.edgecolor": "none", "axes.titlesize": 14, "font.family": "sans-serif"})

    def plot(self, plot_data, report_data):
        original_img, yolo_overlay, cam_overlays = plot_data
        fig, axes = matplotlib.pyplot.subplots(2, 3, figsize=(13, 10))
        fig.subplots_adjust(wspace=0.1, hspace=0.25, top=0.88, bottom=0.05, left=0.05, right=0.95)
        fig.suptitle(f"Dual Model Diagnosis: {report_data['image_id']}", fontsize=20, weight='bold')

        #row1
        axes[0, 0].imshow(original_img); 
        axes[0, 0].set_title("Processed X-Ray Image", weight='bold')

        axes[0, 1].imshow(yolo_overlay); 
        axes[0, 1].set_title("Segmentation", weight='bold')
        self.yolo_annotations(axes[0, 1], report_data)

        self.clinical_summary(axes[0, 2], report_data)

        #row2
        axes[1, 0].imshow(cam_overlays["Grad-CAM"]); axes[1, 0].set_title("Grad-CAM", weight='bold')

        axes[1, 1].imshow(cam_overlays["Grad-CAM++"]); axes[1, 1].set_title("Grad-CAM++", weight='bold')
        
        axes[1, 2].imshow(cam_overlays["Layer-CAM"]); axes[1, 2].set_title("Layer-CAM", weight='bold')

        for axis in axes.flat: axis.axis("off")
        matplotlib.pyplot.show()

    def yolo_annotations(self, ax, data):
        if data['yolo_label'] == "Tumor":
            detail = data['yolo_detail']
            conf = data['yolo_conf']
            color = '#C70039' if 'malignant' in detail.lower() else '#3498DB'
            ax.text(15, 15, f"{detail}\n(conf: {conf:.2f})", color="white", fontsize=11, weight='bold', bbox=dict(boxstyle="round,pad=0.4", fc=color, ec='none', alpha=0.9), va='top', ha='left')

    def clinical_summary(self, ax, data):
        ax.set_title("Diagnostic Summary", weight='bold'); ax.axis("off"); ax.axhline(0.99, color='#E0E0E0', lw=2)
        y = 0.90

        #row1 
        ax.text(0, y, "1. Segmentation", fontsize=12, weight='bold', ha='left')
        yolo_output = f"Output: {data['yolo_detail']}" + (f" (conf: {data['yolo_conf']:.2f})" if data['yolo_label'] == "Tumor" else "")
        ax.text(0.02, y - 0.09, yolo_output, fontsize=11, ha='left', va='top')
        y -= 0.22; ax.axhline(y + 0.05, color='#F0F0F0', ls='--')

        #row2
        ax.text(0, y, "2. Classification", fontsize=12, weight='bold', ha='left')
        classification_output = data['cls_prob'] if data['cls_label'] == "Tumor" else 1 - data['cls_prob']
        ax.text(0.02, y - 0.09, f"Output: {data['cls_label']} (conf: {classification_output:.2f})", fontsize=11, ha='left', va='top')
        y -= 0.22; ax.axhline(y + 0.05, color='#F0F0F0', ls='--')

        #row3
        ax.text(0, y, "System Agreement", fontsize=12, weight='bold', ha='left')
        y -= 0.09
        self.agreement(ax, y, "Presence Agreement:", data['presence_agree'])
        y -= 0.08
        self.agreement(ax, y, "Location Agreement:", data['location_agree'])

        #row4
        summary = data['final_summary']
        ax.add_patch(matplotlib.pyplot.Rectangle((0, 0), 1, 0.15, fc=summary['color'], ec='none', alpha=0.1, transform=ax.transAxes))
        ax.text(0.5, 0.075, summary['text'], fontsize=14, weight='bold', ha='center', va='center', color=summary['color'])


    def agreement(self, ax, y, label, value):
        ax.text(0.02, y, label, fontsize=11, ha='left', va='top')
        if value is None: text, color = "No Tumour", "gray"
        elif value: text, color = "TRUE", "green"
        else: text, color = "FALSE", "red"
        ax.text(0.98, y, text, fontsize=11, weight='bold', ha='right', va='top', color=color)

class App:
    def __init__(self, root, yolo, clf, cams):
        self.root, self.yolo, self.clf, self.cams = root, yolo, clf, cams
        self.visualiser = Visualiser()
        self.sections()

    def sections(self):
        main = ttk.Frame(self.root, padding="10")
        main.pack(fill=tkinter.BOTH, expand=True)

        img_frame = ttk.LabelFrame(main, text="1. Image Selection", padding="10")
        img_frame.pack(fill=tkinter.X, pady=5)
        self.image_path_var = tkinter.StringVar()
        ttk.Label(img_frame, text="Image Path:").pack(side=tkinter.LEFT, padx=(0, 5))
        ttk.Entry(img_frame, textvariable=self.image_path_var).pack(side=tkinter.LEFT, fill=tkinter.X, expand=True)
        ttk.Button(img_frame, text="Browse…", command=self.input_image).pack(side=tkinter.LEFT, padx=5)
        
        ctx_frame = ttk.LabelFrame(main, text="2. Context Configuration", padding="10")
        ctx_frame.pack(fill=tkinter.X, pady=5)
        self.age_var = tkinter.IntVar(value=40)
        self.gender_var = tkinter.StringVar(value='F')
        self.view_var = tkinter.StringVar(value='frontal')

        age_row = ttk.Frame(ctx_frame)
        age_row.pack(fill=tkinter.X, pady=2, padx=5)
        ttk.Label(age_row, text="Age:", width=8).pack(side=tkinter.LEFT)
        ttk.Entry(age_row, textvariable=self.age_var, width=10).pack(side=tkinter.LEFT)

        gender_row = ttk.Frame(ctx_frame)
        gender_row.pack(fill=tkinter.X, pady=2, padx=5)
        ttk.Label(gender_row, text="Gender:", width=8).pack(side=tkinter.LEFT)
        gender_radios = self.input_context(gender_row, [("F", "F"), ("M", "M")], self.gender_var)
        gender_radios.pack(side=tkinter.LEFT)

        view_row = ttk.Frame(ctx_frame)
        view_row.pack(fill=tkinter.X, pady=2, padx=5)
        ttk.Label(view_row, text="View:", width=8).pack(side=tkinter.LEFT)
        view_radios = self.input_context(view_row, [("Frontal", "frontal"), ("Lateral", "lateral"), ("Oblique", "oblique")], self.view_var)
        view_radios.pack(side=tkinter.LEFT)

        ttk.Button(main, text="Run Visualisation", command=self.run_visualisation).pack(fill=tkinter.X, pady=20, ipady=5)


    def input_context(self, parent, options, variable):
        group = ttk.Frame(parent)
        for text, value in options:
            ttk.Radiobutton(group, text=text, variable=variable, value=value).pack(side=tkinter.LEFT, padx=5)
        return group
    
    def input_image(self):
        path = filedialog.askopenfilename(title="Choose X-ray image", filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")])
        if path: self.image_path_var.set(path)

    def run_visualisation(self):
        path = Path(self.image_path_var.get().strip())
        display_rgb, img_tensor = logic.prepare_display_and_tensor(path)

        classifier_result = self._run_classifier(img_tensor)
        yolo_result = self._run_yolo(path)
        cam_maps = self._generate_cams(img_tensor)

        agreement = self._calculate_agreement(classifier_result['label'], yolo_result['label'], yolo_result['mask'], cam_maps)

        report = self._create_report(path.name, classifier_result, yolo_result, agreement)
        plot_data = self._prepare_plot_data(display_rgb, yolo_result['mask'], cam_maps)

        self.visualiser.plot(plot_data, report)

    def _run_classifier(self, img_tensor):
        age, gender, view = self.age_var.get(), self.gender_var.get(), self.view_var.get()
        age_z = (age - 35.57109479305741) / (20.997566907248853 or 1.0)
        view_vec = [1.0, 0.0, 0.0]
        if view == 'lateral': view_vec = [0.0, 1.0, 0.0]
        elif view == 'oblique': view_vec = [0.0, 0.0, 1.0]
        ctx_arr = numpy.array([age_z, 1.0 if gender == 'F' else 0.0] + view_vec, dtype=numpy.float32)[None, :]
        ctx_tensor = torch.from_numpy(ctx_arr).to(torch.device("cpu"))
        
        with torch.no_grad():
            fused, _, _ = self.clf(img_tensor, ctx_tensor)
            prob = torch.sigmoid(fused).item()
        
        label = "Tumor" if prob >= 0.5 else "No Tumor"
        return {'prob': prob, 'label': label}

    def _run_yolo(self, image_path):
        pred = self.yolo.predict(str(image_path), imgsz=640, conf=0.001, verbose=False, device=str(torch.device("cpu")))[0]
        
        mask_union = numpy.zeros((640, 640), dtype=numpy.uint8)
        best_conf, best_detail = -1.0, None

        if pred.boxes:
            for i in range(len(pred.boxes)):
                cls_name = pred.names[int(pred.boxes.cls[i])]
                if logic.is_tumor(cls_name):
                    if pred.masks:
                        m = (cv2.resize(pred.masks.data[i].cpu().numpy(), (640, 640), interpolation=cv2.INTER_NEAREST) > 0.5)
                    else: 
                        m = numpy.zeros((640, 640), numpy.uint8)
                        x1, y1, x2, y2 = pred.boxes.xyxy[i].cpu().numpy().astype(int)
                        cv2.rectangle(m, (x1, y1), (x2, y2), 1, -1)
                    mask_union |= m.astype(numpy.uint8)
                    
                    conf = float(pred.boxes.conf[i])
                    if conf > best_conf:
                        best_conf = conf
                        best_detail = cls_name.replace("_", " ").title()

        if best_detail:
            return {'label': "Tumor", 'detail': best_detail, 'conf': best_conf, 'mask': mask_union}
        return {'label': "No Tumor", 'detail': "No Tumor", 'conf': 0.0, 'mask': mask_union}

    def _generate_cams(self, img_tensor):
        targets = [ClassifierOutputTarget(0)]
        cam_maps = {}
        for name, cam_method in self.cams.items():
            raw_map = cam_method(input_tensor=img_tensor, targets=targets)[0]
            cam_maps[name] = logic.norm01(cv2.resize(raw_map, (640, 640)))
        return cam_maps
    
    def _calculate_agreement(self, cls_label, yolo_label, yolo_mask, cam_maps):
        presence = (cls_label == yolo_label)
        location = None
        
        if presence and cls_label == "Tumor":
            location = any(numpy.logical_and(cam >= 0.5, yolo_mask.astype(bool)).any() for cam in cam_maps.values())
            summary = {'text': 'CONCLUSION: Tumor Detected', 'color': '#2ECC71'} if location else {'text': 'WARNING: Location Inconsistent', 'color': '#F39C12'}
        elif presence: 
            summary = {'text': 'CONCLUSION: No Tumor', 'color': '#2ECC71'}
        else: 
            summary = {'text': 'WARNING: Model Disagreement', 'color': '#E74C3C'}
            
        return {'presence': presence, 'location': location, 'summary': summary}

    def _prepare_plot_data(self, display_rgb, yolo_mask, cam_maps):
        yolo_overlay = display_rgb.copy()
        yolo_overlay[yolo_mask > 0] = (yolo_overlay[yolo_mask > 0] * 0.5 + numpy.array([0, 0, 255]) * 0.5).astype(numpy.uint8)
        cam_overlays = {name: logic.overlay_cam_on_image(display_rgb, cam) for name, cam in cam_maps.items()}
        return (display_rgb, yolo_overlay, cam_overlays)

    def _create_report(self, image_id, cls, yolo, agree):
        return {'image_id': image_id,'yolo_label': yolo['label'], 'yolo_detail': yolo['detail'], 'yolo_conf': yolo['conf'],'cls_label': cls['label'], 'cls_prob': cls['prob'],'presence_agree': agree['presence'], 'location_agree': agree['location'], 'final_summary': agree['summary']}

def main():
    model = logic.EfficientNetB3().to(torch.device("cpu")).eval()
    ckpt = torch.load(Path.cwd() / "models/EfficientNet-B3.pt", map_location=torch.device("cpu"))
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    yolo = YOLO(str(Path.cwd() / "models/YOLOv8.pt"))

    target_layer = logic.get_target_conv(model)
    wrapped_model = logic._ImgOnlyWrapper(model).to(torch.device("cpu")).eval()
    cams = {"Grad-CAM": GradCAM(model=wrapped_model, target_layers=[target_layer]),"Grad-CAM++": GradCAMPlusPlus(model=wrapped_model, target_layers=[target_layer]),"Layer-CAM": LayerCAM(model=wrapped_model, target_layers=[target_layer])}
    root = tkinter.Tk()
    App(root, yolo, model, cams)
    root.mainloop()

if __name__ == "__main__":
    main()