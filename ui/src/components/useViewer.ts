import { useRef, useEffect } from "react";
import React from "react";

import { Niivue, NVController } from "@niivue/niivue";



export default function useViewer(niftiUrl: string, onNodeClick: (node: number[]) => void) {
    let defaults = {
        loadingText: "there are no images",
        show3Dcrosshair: true,
      };
      const canvas = useRef();
      const nv = new Niivue(defaults);
      const connectomeUrl = "/connectome.json";
      const currentNode = useRef<number[]>([0, 0, 0]);

      function updateNode(XYZmm: number[]) {
        nv.meshes[0].nodes[0].x = XYZmm[0];
        nv.meshes[0].nodes[0].y = XYZmm[1];
        nv.meshes[0].nodes[0].z = XYZmm[2];
        nv.meshes[0].nodes[0].sizeValue = 2;
        nv.meshes[0].updateMesh(nv.gl);
        nv.updateGLVolume();
      }

      useEffect(() => {
        if (!canvas.current) return;
        async function load() {
        const volumeList = [
          {
            url: niftiUrl,
            colormap: "ct_bones", // see: https://niivue.github.io/niivue/colormaps.html,
            opacity: 1.0,
          },
        ];
          nv.attachToCanvas(canvas.current);
          nv.setSliceType(nv.sliceTypeMultiplanar);
          nv.setMultiplanarLayout(2);
          nv.opts.multiplanarForceRender = true;
          nv.graph.opacity = 1.0;
          nv.opts.meshXRay = 0.6;
          // nv.setClipPlane([0.12, 180, 0]);
          await nv.loadVolumes(volumeList);
          await nv.loadFreeSurferConnectomeFromUrl(connectomeUrl);
          nv.setHighResolutionCapable(true);
          nv.onMouseUp = (uiData) => {
            // @ts-ignore: fracPos is not defined in the type
            if (uiData.fracPos[0] < 0) return; //not on volume
            // @ts-ignore: fracPos is not defined in the type
            const norm = (uiData.fracPos[0] - currentNode.current[0])**2 + (uiData.fracPos[1] - currentNode.current[1])**2 + (uiData.fracPos[2] - currentNode.current[2])**2;
            if (norm < 0.00001) {console.log("same node"); return;};
            // @ts-ignore: fracPos is not defined in the type
            let XYZmmVec = nv.frac2mm(uiData.fracPos);
            let XYZmm = [XYZmmVec[0], XYZmmVec[1], XYZmmVec[2]];
            updateNode(XYZmm);
            onNodeClick(XYZmm);
            // @ts-ignore: fracPos is not defined in the type
            currentNode.current = uiData.fracPos;
          }
        }
        load();
      }, [canvas]);

      return { canvas };
}
