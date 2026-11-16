"""Live diffraction-efficiency analysis with a Thorlabs camera.

    python main.py                 # pick a camera in the GUI
    python main.py --simulate      # start straight away with the simulated camera
    python main.py --image x.tif   # simulated camera replaying an image file
"""
import argparse
import sys

from PySide6 import QtCore, QtWidgets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--simulate", action="store_true", help="connect to the simulated camera on start")
    ap.add_argument("--image", help="image file replayed by the simulated camera (implies --simulate)")
    ap.add_argument("--screenshot", help=argparse.SUPPRESS)  # self-test: save a screenshot after 4 s and quit
    args = ap.parse_args()

    sim_image = None
    if args.image:
        import numpy as np
        from PIL import Image
        sim_image = np.array(Image.open(args.image))

    from lde.app import SIM_LABEL, MainWindow

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(sim_image=sim_image)
    win.show()
    if args.simulate or args.image:
        win.cam_combo.setCurrentIndex(win.cam_combo.findText(SIM_LABEL))
        win.connect_camera()
    elif win.cam_combo.count() > 1:  # a real camera is present: connect to the first one
        win.connect_camera()
    if args.screenshot:
        def shoot():
            win.grab().save(args.screenshot)
            win.close()
        QtCore.QTimer.singleShot(4000, shoot)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
