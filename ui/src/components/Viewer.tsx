import React from "react";
import useViewer from "./useViewer";


interface ViewerProps {
  niftiUrl: string;
  onNodeClick: (node: number[]) => void;
}

const Viewer = ({ niftiUrl, onNodeClick }: ViewerProps) => {

  const { canvas } = useViewer(niftiUrl, onNodeClick);

  return <div style={{ flexBasis: "50%" }}>
    <canvas ref={canvas} height="100%"/>
  </div>;
};
export default Viewer
