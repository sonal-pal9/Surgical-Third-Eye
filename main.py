import cv2
import matplotlib.pyplot as plt



VIDEO_PATH = "capture1.avi"
N = 20  
CURRENT_PROCEDURE = "laparoscopy"

PROCEDURE_PROFILES = {
    "laparoscopy": {
        "critical_radius": 120,
        "speed_th": 45
    }
}

profile = PROCEDURE_PROFILES[CURRENT_PROCEDURE]


RISK_COLORS = {
    "SAFE": (0, 255, 0),
    "LOW": (173, 255, 47),
    "MODERATE": (0, 165, 255),
    "HIGH": (0, 0, 255),
    "CRITICAL": (0, 0, 139)
}


report = {
    "good_vis": 0,
    "moderate_vis": 0,
    "poor_vis": 0,
    "tool_collision": 0,
    "critical_region": 0,
    "fast_tool": 0,
    "reduced_visibility_risk": 0,
    "total_risk_events": 0,
    "total_frames": 0
}

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    print("Video opened:", cap.isOpened())

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        report["total_frames"] += 1
        
        # Phase 1: Visibility

        visibility, reasons, confidence = assess_visibility(frame)

        if visibility == "GOOD":
            report["good_vis"] += 1
        elif visibility == "MODERATE":
            report["moderate_vis"] += 1
        else:
            report["poor_vis"] += 1
        
        # Phase 2: Tool Detection

        tools = detect_tools(frame)
        safety_flags = []

        safety = assess_tool_safety(
            tools,
            threshold=profile["critical_radius"]
        )
     
        if safety == "RISK":
            safety_flags.append("TOOL_PROXIMITY")
            report["tool_collision"] += 1
            report["total_risk_events"] += 1
            if visibility != "GOOD":
                report["reduced_visibility_risk"] += 1

        # Phase 3: Critical Region

        critical_alert = critical_region_detected(
            tools,
            profile["critical_radius"]
        )

        if critical_alert:
            safety_flags.append("CRITICAL_REGION")
            report["critical_region"] += 1
            report["total_risk_events"] += 1
            if visibility != "GOOD":
                report["reduced_visibility_risk"] += 1

        # Phase 4: Tool Speed

        speed = tool_speed(tools)

        if speed > profile["speed_th"]:
            safety_flags.append("HIGH_TOOL_SPEED")
            report["fast_tool"] += 1
            report["total_risk_events"] += 1
            if visibility != "GOOD":
                report["reduced_visibility_risk"] += 1
        
         # Phase 5: Contextual Risk
         
        risk_level = contextual_risk(
            visibility,
            safety_flags,
            confidence
        )


        color = (0, 255, 0) if visibility == "GOOD" else \
                (0, 165, 255) if visibility == "MODERATE" else (0, 0, 255)

        vis_label = f"Visibility: {visibility}"
        if reasons:
            vis_label += f" ({', '.join(reasons)})"
        vis_label += f" | {confidence}%"

        cv2.putText(frame, vis_label, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        draw_tool_alerts(frame, tools, safety)

        cv2.putText(frame,
                    f"Contextual Risk: {risk_level}",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    RISK_COLORS[risk_level],
                    3)

        cv2.putText(frame,
                    f"Procedure: {CURRENT_PROCEDURE.upper()}",
                    (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        y = 160
        for flag in safety_flags:
            cv2.putText(frame,
                        f"• {flag}",
                        (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2)
            y += 25

        if critical_alert:
            cv2.putText(frame,
                        "CRITICAL REGION ALERT!",
                        (30, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3)

        if frame_count % N == 0:
            plt.figure(figsize=(5, 5))
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

    cap.release()
    print("Processing complete.")
    print("Final Report:", report)


if __name__ == "__main__":
    main()
