import { PatientWorkspace } from "@/components/patient-workspace";

type PatientPageProps = {
  params: {
    patientId: string;
  };
};

export default function PatientPage({ params }: PatientPageProps) {
  return <PatientWorkspace patientId={decodeURIComponent(params.patientId)} />;
}