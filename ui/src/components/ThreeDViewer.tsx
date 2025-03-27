import React from "react";
import useViewer from "../hooks/useViewer";


interface ThreeDViewerProps {
  niftiUrl: string;
  onNodeClick: (node: number[]) => void;
}

const ThreeDViewer = ({ niftiUrl, onNodeClick }: ThreeDViewerProps) => {

  const { canvas } = useViewer(niftiUrl, onNodeClick);

  return <div style={{ flexBasis: "50%" }}>
    <canvas ref={canvas} height="100%"/>
  </div>;
};
export default ThreeDViewer
